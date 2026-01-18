from __future__ import annotations
import base64
import collections.abc
import logging
import os
import typing as t
from spnego._context import (
from spnego._credential import Credential, CredentialCache, Password, unify_credentials
from spnego.channel_bindings import GssChannelBindings
from spnego.exceptions import (
from spnego.exceptions import WinError as NativeError
from spnego.iov import BufferType, IOVBuffer, IOVResBuffer
class SSPIProxy(ContextProxy):
    """SSPI proxy class for pure SSPI on Windows.

    This proxy class for SSPI exposes this library into a common interface for SPNEGO authentication. This context
    uses compiled C code to interface directly into the SSPI functions on Windows to provide a native SPNEGO
    implementation.
    """

    def __init__(self, username: str | Credential | list[Credential] | None=None, password: str | None=None, hostname: str | None=None, service: str | None=None, channel_bindings: GssChannelBindings | None=None, context_req: ContextReq=ContextReq.default, usage: str='initiate', protocol: str='negotiate', options: NegotiateOptions=NegotiateOptions.none, **kwargs: t.Any) -> None:
        if not HAS_SSPI:
            raise ImportError('SSPIProxy requires the Windows only sspilib python package')
        credentials = unify_credentials(username, password)
        super(SSPIProxy, self).__init__(credentials, hostname, service, channel_bindings, context_req, usage, protocol, options)
        self._native_channel_bindings: sspilib.SecChannelBindings | None
        if channel_bindings:
            self._native_channel_bindings = self._get_native_bindings(channel_bindings)
        else:
            self._native_channel_bindings = None
        self._block_size = 0
        self._max_signature = 0
        self._security_trailer = 0
        self._complete = False
        self._context: sspilib.raw.CtxtHandle | None = None
        self.__seq_num = 0
        sspi_credential = kwargs.get('_sspi_credential', None)
        if not sspi_credential:
            try:
                principal = self.spn if usage == 'accept' else None
                sspi_credential = _get_sspi_credential(principal, protocol, usage, credentials)
            except NativeError as win_err:
                raise SpnegoError(base_error=win_err, context_msg='Getting SSPI credential') from win_err
        self._credential = sspi_credential

    @classmethod
    def available_protocols(cls, options: NegotiateOptions | None=None) -> list[str]:
        return _available_protocols()

    @property
    def client_principal(self) -> str | None:
        if self.usage == 'accept':
            names = sspilib.raw.query_context_attributes(t.cast(sspilib.raw.CtxtHandle, self._context), sspilib.raw.SecPkgContextNames)
            return names.username
        else:
            return None

    @property
    def complete(self) -> bool:
        return self._complete

    @property
    def negotiated_protocol(self) -> str | None:
        package_info = sspilib.raw.query_context_attributes(t.cast(sspilib.raw.CtxtHandle, self._context), sspilib.raw.SecPkgContextPackageInfo)
        return package_info.name.lower()

    @property
    @wrap_system_error(NativeError, 'Retrieving session key')
    def session_key(self) -> bytes:
        session_key = sspilib.raw.query_context_attributes(t.cast(sspilib.raw.CtxtHandle, self._context), sspilib.raw.SecPkgContextSessionKey)
        return session_key.session_key

    def new_context(self) -> SSPIProxy:
        return SSPIProxy(hostname=self._hostname, service=self._service, channel_bindings=self.channel_bindings, context_req=self.context_req, usage=self.usage, protocol=self.protocol, options=self.options, _sspi_credential=self._credential)

    @wrap_system_error(NativeError, 'Processing security token')
    def step(self, in_token: bytes | None=None, *, channel_bindings: GssChannelBindings | None=None) -> bytes | None:
        if not self._is_wrapped:
            log.debug('SSPI step input: %s', base64.b64encode(in_token or b'').decode())
        sec_tokens: list[sspilib.raw.SecBuffer] = []
        if in_token:
            in_token = bytearray(in_token)
            sec_tokens.append(sspilib.raw.SecBuffer(in_token, sspilib.raw.SecBufferType.SECBUFFER_TOKEN))
        native_channel_bindings: sspilib.SecChannelBindings | None
        if channel_bindings:
            native_channel_bindings = self._get_native_bindings(channel_bindings)
        else:
            native_channel_bindings = self._native_channel_bindings
        if native_channel_bindings:
            sec_tokens.append(native_channel_bindings.dangerous_get_sec_buffer())
        in_buffer: sspilib.raw.SecBufferDesc | None = None
        if sec_tokens:
            in_buffer = sspilib.raw.SecBufferDesc(sec_tokens)
        out_buffer = sspilib.raw.SecBufferDesc([sspilib.raw.SecBuffer(None, sspilib.raw.SecBufferType.SECBUFFER_TOKEN)])
        context_req: int
        res: sspilib.raw.InitializeContextResult | sspilib.raw.AcceptContextResult
        if self.usage == 'initiate':
            context_req = self._context_req | sspilib.IscReq.ISC_REQ_ALLOCATE_MEMORY
            res = sspilib.raw.initialize_security_context(credential=self._credential, context=self._context, target_name=self.spn or '', context_req=context_req, target_data_rep=sspilib.raw.TargetDataRep.SECURITY_NATIVE_DREP, input_buffers=in_buffer, output_buffers=out_buffer)
            status = res.status
            self._context = res.context
        else:
            context_req = self._context_req | sspilib.AscReq.ASC_REQ_ALLOCATE_MEMORY
            res = sspilib.raw.accept_security_context(credential=self._credential, context=self._context, input_buffers=in_buffer, context_req=context_req, target_data_rep=sspilib.raw.TargetDataRep.SECURITY_NATIVE_DREP, output_buffers=out_buffer)
            status = res.status
            self._context = res.context
        out_token = out_buffer[0].data or None
        self._context_attr = int(res.attributes)
        if status == sspilib.raw.NtStatus.SEC_E_OK:
            self._complete = True
            attr_sizes = sspilib.raw.query_context_attributes(self._context, sspilib.raw.SecPkgContextSizes)
            self._block_size = attr_sizes.block_size
            self._max_signature = attr_sizes.max_signature
            self._security_trailer = attr_sizes.security_trailer
        if not self._is_wrapped:
            log.debug('SSPI step output: %s', base64.b64encode(out_token or b'').decode())
        return out_token

    def query_message_sizes(self) -> SecPkgContextSizes:
        if not self._security_trailer:
            raise NoContextError(context_msg='Cannot get message sizes until context has been established')
        return SecPkgContextSizes(header=self._security_trailer)

    def wrap(self, data: bytes, encrypt: bool=True, qop: int | None=None) -> WrapResult:
        res = self.wrap_iov([BufferType.header, data, BufferType.padding], encrypt=encrypt, qop=qop)
        return WrapResult(data=b''.join([r.data for r in res.buffers if r.data]), encrypted=res.encrypted)

    @wrap_system_error(NativeError, 'Wrapping IOV buffer')
    def wrap_iov(self, iov: collections.abc.Iterable[IOV], encrypt: bool=True, qop: int | None=None) -> IOVWrapResult:
        qop = qop or 0
        if encrypt and qop & sspilib.raw.QopFlags.SECQOP_WRAP_NO_ENCRYPT:
            raise ValueError('Cannot set qop with SECQOP_WRAP_NO_ENCRYPT and encrypt=True')
        elif not encrypt:
            qop |= sspilib.raw.QopFlags.SECQOP_WRAP_NO_ENCRYPT
        buffers = self._build_iov_list(iov, self._convert_iov_buffer)
        iov_buffer = sspilib.raw.SecBufferDesc(buffers)
        sspilib.raw.encrypt_message(t.cast(sspilib.raw.CtxtHandle, self._context), qop=qop, message=iov_buffer, seq_no=self._seq_num)
        return IOVWrapResult(buffers=_create_iov_result(iov_buffer), encrypted=encrypt)

    def wrap_winrm(self, data: bytes) -> WinRMWrapResult:
        iov = self.wrap_iov([BufferType.header, data]).buffers
        header = iov[0].data or b''
        enc_data = iov[1].data or b''
        return WinRMWrapResult(header=header, data=enc_data, padding_length=0)

    def unwrap(self, data: bytes) -> UnwrapResult:
        res = self.unwrap_iov([(BufferType.stream, data), BufferType.data])
        dec_data = res.buffers[1].data or b''
        return UnwrapResult(data=dec_data, encrypted=res.encrypted, qop=res.qop)

    @wrap_system_error(NativeError, 'Unwrapping IOV buffer')
    def unwrap_iov(self, iov: collections.abc.Iterable[IOV]) -> IOVUnwrapResult:
        buffers = self._build_iov_list(iov, self._convert_iov_buffer)
        iov_buffer = sspilib.raw.SecBufferDesc(buffers)
        qop = sspilib.raw.decrypt_message(t.cast(sspilib.raw.CtxtHandle, self._context), iov_buffer, seq_no=self._seq_num)
        encrypted = qop & sspilib.raw.QopFlags.SECQOP_WRAP_NO_ENCRYPT == 0
        return IOVUnwrapResult(buffers=_create_iov_result(iov_buffer), encrypted=encrypted, qop=qop)

    def unwrap_winrm(self, header: bytes, data: bytes) -> bytes:
        iov = self.unwrap_iov([(BufferType.header, header), data]).buffers
        return iov[1].data or b''

    @wrap_system_error(NativeError, 'Signing message')
    def sign(self, data: bytes, qop: int | None=None) -> bytes:
        data = bytearray(data)
        signature = bytearray(self._max_signature)
        iov = sspilib.raw.SecBufferDesc([sspilib.raw.SecBuffer(data, sspilib.raw.SecBufferType.SECBUFFER_DATA), sspilib.raw.SecBuffer(signature, sspilib.raw.SecBufferType.SECBUFFER_TOKEN)])
        sspilib.raw.make_signature(t.cast(sspilib.raw.CtxtHandle, self._context), qop or 0, iov, self._seq_num)
        return iov[1].data

    @wrap_system_error(NativeError, 'Verifying message')
    def verify(self, data: bytes, mic: bytes) -> int:
        data = bytearray(data)
        mic = bytearray(mic)
        iov = sspilib.raw.SecBufferDesc([sspilib.raw.SecBuffer(data, sspilib.raw.SecBufferType.SECBUFFER_DATA), sspilib.raw.SecBuffer(mic, sspilib.raw.SecBufferType.SECBUFFER_TOKEN)])
        return sspilib.raw.verify_signature(t.cast(sspilib.raw.CtxtHandle, self._context), iov, self._seq_num)

    @property
    def _context_attr_map(self) -> list[tuple[ContextReq, int]]:
        attr_map = []
        sspi_req: type[int] | None
        if self.usage == 'initiate':
            attr_map.append((ContextReq.no_integrity, 'REQ_NO_INTEGRITY'))
            sspi_req = sspilib.IscReq
            sspi_prefix = 'ISC'
        else:
            sspi_req = sspilib.AscReq
            sspi_prefix = 'ASC'
        attr_map.extend([(ContextReq.delegate, 'REQ_DELEGATE'), (ContextReq.delegate_policy, 'REQ_DELEGATE'), (ContextReq.mutual_auth, 'REQ_MUTUAL_AUTH'), (ContextReq.replay_detect, 'REQ_REPLAY_DETECT'), (ContextReq.sequence_detect, 'REQ_SEQUENCE_DETECT'), (ContextReq.confidentiality, 'REQ_CONFIDENTIALITY'), (ContextReq.integrity, 'REQ_INTEGRITY'), (ContextReq.dce_style, 'REQ_USE_DCE_STYLE'), (ContextReq.identify, 'REQ_IDENTIFY')])
        attrs = []
        for spnego_flag, gssapi_name in attr_map:
            attrs.append((spnego_flag, getattr(sspi_req, f'{sspi_prefix}_{gssapi_name}')))
        return attrs

    @property
    def _seq_num(self) -> int:
        num = self.__seq_num
        self.__seq_num += 1
        return num

    def _convert_iov_buffer(self, buffer: IOVBuffer) -> sspilib.raw.SecBuffer:
        data = bytearray()
        if isinstance(buffer.data, bytes):
            data = bytearray(buffer.data)
        elif isinstance(buffer.data, int) and (not isinstance(buffer.data, bool)):
            data = bytearray(buffer.data)
        else:
            auto_alloc_size = {BufferType.header: self._security_trailer, BufferType.padding: self._block_size, BufferType.trailer: self._security_trailer}
            alloc = buffer.data
            if alloc is None:
                alloc = buffer.type in auto_alloc_size
            if alloc:
                if buffer.type not in auto_alloc_size:
                    raise ValueError('Cannot auto allocate buffer of type %s.%s' % (type(buffer.type).__name__, buffer.type.name))
                data = bytearray(auto_alloc_size[buffer.type])
        buffer_type = int(buffer.type)
        buffer_flags = 0
        if buffer_type == BufferType.sign_only:
            buffer_type = sspilib.raw.SecBufferType.SECBUFFER_DATA
            buffer_flags = sspilib.raw.SecBufferFlags.SECBUFFER_READONLY_WITH_CHECKSUM
        elif buffer_type == BufferType.data_readonly:
            buffer_type = sspilib.raw.SecBufferType.SECBUFFER_DATA
            buffer_flags = sspilib.raw.SecBufferFlags.SECBUFFER_READONLY
        return sspilib.raw.SecBuffer(data, buffer_type, buffer_flags)

    def _get_native_bindings(self, channel_bindings: GssChannelBindings) -> sspilib.SecChannelBindings:
        """Gets the raw byte value of the SEC_CHANNEL_BINDINGS structure."""
        return sspilib.SecChannelBindings(initiator_addr_type=int(channel_bindings.initiator_addrtype), initiator_addr=channel_bindings.initiator_address, acceptor_addr_type=int(channel_bindings.acceptor_addrtype), acceptor_addr=channel_bindings.acceptor_address, application_data=channel_bindings.application_data)