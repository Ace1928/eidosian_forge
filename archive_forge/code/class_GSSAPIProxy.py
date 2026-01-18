import base64
import copy
import logging
import sys
import typing
from spnego._context import (
from spnego._credential import (
from spnego._text import to_bytes, to_text
from spnego.channel_bindings import GssChannelBindings
from spnego.exceptions import GSSError as NativeError
from spnego.exceptions import (
from spnego.iov import BufferType, IOVBuffer, IOVResBuffer
class GSSAPIProxy(ContextProxy):
    """GSSAPI proxy class for GSSAPI on Linux.

    This proxy class for GSSAPI exposes GSSAPI calls into a common interface for Kerberos authentication. This context
    uses the Python gssapi library to interface with the gss_* calls to provider Kerberos.
    """

    def __init__(self, username: typing.Optional[typing.Union[str, Credential, typing.List[Credential]]]=None, password: typing.Optional[str]=None, hostname: typing.Optional[str]=None, service: typing.Optional[str]=None, channel_bindings: typing.Optional[GssChannelBindings]=None, context_req: ContextReq=ContextReq.default, usage: str='initiate', protocol: str='kerberos', options: NegotiateOptions=NegotiateOptions.none, **kwargs: typing.Any) -> None:
        if not HAS_GSSAPI:
            raise ImportError('GSSAPIProxy requires the Python gssapi library: %s' % GSSAPI_IMP_ERR)
        credentials = unify_credentials(username, password)
        super(GSSAPIProxy, self).__init__(credentials, hostname, service, channel_bindings, context_req, usage, protocol, options)
        self._mech = gssapi.OID.from_int_seq(GSSMech.kerberos.value)
        gssapi_credential = kwargs.get('_gssapi_credential', None)
        if not gssapi_credential:
            try:
                gssapi_credential = _get_gssapi_credential(self._mech, self.usage, credentials=credentials, context_req=context_req)
            except GSSError as gss_err:
                raise SpnegoError(base_error=gss_err, context_msg='Getting GSSAPI credential') from gss_err
        if context_req & ContextReq.no_integrity and self.usage == 'initiate':
            if gssapi_credential is None:
                gssapi_credential = gssapi.Credentials(usage=self.usage, mechs=[self._mech])
            set_cred_option(gssapi.OID.from_int_seq(_GSS_KRB5_CRED_NO_CI_FLAGS_X), gssapi_credential)
        self._credential = gssapi_credential
        self._context: typing.Optional[gssapi.SecurityContext] = None

    @classmethod
    def available_protocols(cls, options: typing.Optional[NegotiateOptions]=None) -> typing.List[str]:
        avail = []
        if not (options and options & NegotiateOptions.wrapping_winrm and (not HAS_IOV)):
            avail.append('kerberos')
        return avail

    @classmethod
    def iov_available(cls) -> bool:
        return HAS_IOV

    @property
    def client_principal(self) -> typing.Optional[str]:
        if self._context and self.usage == 'accept':
            return to_text(self._context.initiator_name).rstrip('\x00')
        else:
            return None

    @property
    def complete(self) -> bool:
        return self._context is not None and self._context.complete

    @property
    def negotiated_protocol(self) -> typing.Optional[str]:
        return 'kerberos'

    @property
    @wrap_system_error(NativeError, 'Retrieving session key')
    def session_key(self) -> bytes:
        if self._context:
            return inquire_sec_context_by_oid(self._context, gssapi.OID.from_int_seq(_GSS_C_INQ_SSPI_SESSION_KEY))[0]
        else:
            raise NoContextError(context_msg='Retrieving session key failed as no context was initialized')

    def new_context(self) -> 'GSSAPIProxy':
        return GSSAPIProxy(hostname=self._hostname, service=self._service, channel_bindings=self.channel_bindings, context_req=self.context_req, usage=self.usage, protocol=self.protocol, options=self.options, _gssapi_credential=self._credential)

    @wrap_system_error(NativeError, 'Processing security token')
    def step(self, in_token: typing.Optional[bytes]=None, *, channel_bindings: typing.Optional[GssChannelBindings]=None) -> typing.Optional[bytes]:
        if not self._is_wrapped:
            log.debug('GSSAPI step input: %s', base64.b64encode(in_token or b'').decode())
        if not self._context:
            context_kwargs: typing.Dict[str, typing.Any] = {}
            channel_bindings = channel_bindings or self.channel_bindings
            if channel_bindings:
                context_kwargs['channel_bindings'] = ChannelBindings(initiator_address_type=channel_bindings.initiator_addrtype, initiator_address=channel_bindings.initiator_address, acceptor_address_type=channel_bindings.acceptor_addrtype, acceptor_address=channel_bindings.acceptor_address, application_data=channel_bindings.application_data)
            if self.usage == 'initiate':
                spn = '%s@%s' % (self._service or 'host', self._hostname or 'unspecified')
                context_kwargs['name'] = gssapi.Name(spn, name_type=gssapi.NameType.hostbased_service)
                context_kwargs['mech'] = self._mech
                context_kwargs['flags'] = self._context_req
            self._context = gssapi.SecurityContext(creds=self._credential, usage=self.usage, **context_kwargs)
        out_token = self._context.step(in_token)
        try:
            self._context_attr = int(self._context.actual_flags)
        except gss_errors.MissingContextError:
            if self._context.complete:
                raise
        if not self._is_wrapped:
            log.debug('GSSAPI step output: %s', base64.b64encode(out_token or b'').decode())
        return out_token

    @wrap_system_error(NativeError, 'Getting context sizes')
    def query_message_sizes(self) -> SecPkgContextSizes:
        if not self._context:
            raise NoContextError(context_msg='Cannot get message sizes until context has been established')
        iov = GSSIOV(IOVBufferType.header, b'', std_layout=False)
        wrap_iov_length(self._context, iov)
        return SecPkgContextSizes(header=len(iov[0].value or b''))

    @wrap_system_error(NativeError, 'Wrapping data')
    def wrap(self, data: bytes, encrypt: bool=True, qop: typing.Optional[int]=None) -> WrapResult:
        if not self._context:
            raise NoContextError(context_msg='Cannot wrap until context has been established')
        res = gssapi.raw.wrap(self._context, data, confidential=encrypt, qop=qop)
        return WrapResult(data=res.message, encrypted=res.encrypted)

    @wrap_system_error(NativeError, 'Wrapping IOV buffer')
    def wrap_iov(self, iov: typing.Iterable[IOV], encrypt: bool=True, qop: typing.Optional[int]=None) -> IOVWrapResult:
        if not self._context:
            raise NoContextError(context_msg='Cannot wrap until context has been established')
        buffers = self._build_iov_list(iov, self._convert_iov_buffer)
        iov_buffer = GSSIOV(*buffers, std_layout=False)
        encrypted = wrap_iov(self._context, iov_buffer, confidential=encrypt, qop=qop)
        return IOVWrapResult(buffers=_create_iov_result(iov_buffer), encrypted=encrypted)

    def wrap_winrm(self, data: bytes) -> WinRMWrapResult:
        iov = self.wrap_iov([BufferType.header, data, BufferType.padding]).buffers
        header = iov[0].data or b''
        enc_data = iov[1].data or b''
        padding = iov[2].data or b''
        return WinRMWrapResult(header=header, data=enc_data + padding, padding_length=len(padding))

    @wrap_system_error(NativeError, 'Unwrapping data')
    def unwrap(self, data: bytes) -> UnwrapResult:
        if not self._context:
            raise NoContextError(context_msg='Cannot unwrap until context has been established')
        res = gssapi.raw.unwrap(self._context, data)
        return UnwrapResult(data=res.message, encrypted=res.encrypted, qop=res.qop)

    @wrap_system_error(NativeError, 'Unwrapping IOV buffer')
    def unwrap_iov(self, iov: typing.Iterable[IOV]) -> IOVUnwrapResult:
        if not self._context:
            raise NoContextError(context_msg='Cannot unwrap until context has been established')
        buffers = self._build_iov_list(iov, self._convert_iov_buffer)
        iov_buffer = GSSIOV(*buffers, std_layout=False)
        res = unwrap_iov(self._context, iov_buffer)
        return IOVUnwrapResult(buffers=_create_iov_result(iov_buffer), encrypted=res.encrypted, qop=res.qop)

    def unwrap_winrm(self, header: bytes, data: bytes) -> bytes:
        if not self._context:
            raise NoContextError(context_msg='Cannot unwrap until context has been established')
        sasl_desc = _gss_sasl_description(self._context.mech)
        if sasl_desc and sasl_desc == b'Kerberos 5 GSS-API Mechanism':
            iov = self.unwrap_iov([(IOVBufferType.header, header), data, IOVBufferType.data]).buffers
            return iov[1].data or b''
        else:
            return self.unwrap(header + data).data

    @wrap_system_error(NativeError, 'Signing message')
    def sign(self, data: bytes, qop: typing.Optional[int]=None) -> bytes:
        if not self._context:
            raise NoContextError(context_msg='Cannot sign until context has been established')
        return gssapi.raw.get_mic(self._context, data, qop=qop)

    @wrap_system_error(NativeError, 'Verifying message')
    def verify(self, data: bytes, mic: bytes) -> int:
        if not self._context:
            raise NoContextError(context_msg='Cannot verify until context has been established')
        return gssapi.raw.verify_mic(self._context, data, mic)

    @property
    def _context_attr_map(self) -> typing.List[typing.Tuple[ContextReq, int]]:
        attr_map = [(ContextReq.delegate, 'delegate_to_peer'), (ContextReq.mutual_auth, 'mutual_authentication'), (ContextReq.replay_detect, 'replay_detection'), (ContextReq.sequence_detect, 'out_of_sequence_detection'), (ContextReq.confidentiality, 'confidentiality'), (ContextReq.integrity, 'integrity'), (ContextReq.dce_style, 'dce_style'), (ContextReq.identify, 'identify'), (ContextReq.delegate_policy, 'ok_as_delegate')]
        attrs = []
        for spnego_flag, gssapi_name in attr_map:
            if hasattr(gssapi.RequirementFlag, gssapi_name):
                attrs.append((spnego_flag, getattr(gssapi.RequirementFlag, gssapi_name)))
        return attrs

    def _convert_iov_buffer(self, buffer: IOVBuffer) -> 'GSSIOVBuffer':
        buffer_data = None
        buffer_alloc = False
        if isinstance(buffer.data, bytes):
            buffer_data = buffer.data
        elif isinstance(buffer.data, bool):
            buffer_alloc = buffer.data
        elif isinstance(buffer.data, int):
            buffer_data = b'\x00' * buffer.data
        else:
            auto_alloc = [BufferType.header, BufferType.padding, BufferType.trailer]
            buffer_alloc = buffer.type in auto_alloc
        buffer_type = buffer.type
        if buffer.type == BufferType.data_readonly:
            buffer_type = BufferType.empty
        return GSSIOVBuffer(IOVBufferType(buffer_type), buffer_alloc, buffer_data)