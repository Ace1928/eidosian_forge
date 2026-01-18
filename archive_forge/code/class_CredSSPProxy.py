import base64
import hashlib
import logging
import os
import re
import shutil
import ssl
import struct
import tempfile
import typing
import spnego
from spnego._context import (
from spnego._credential import Credential, Password, unify_credentials
from spnego._credssp_structures import (
from spnego._text import to_text
from spnego.channel_bindings import GssChannelBindings
from spnego.exceptions import (
from spnego.tls import (
class CredSSPProxy(ContextProxy):
    """CredSSP proxy class CredSSP authentication.

    This proxy class for CredSSP can be used to exchange CredSSP tokens.
    It uses the NegotiateProxy provider for the underlying authentication but
    exchanges the tokens in the exchange required by CredSSP. The main
    advantage of CredSSP is that it allows you to delegate the user's
    credentials to the server.

    The acceptor logic is mostly done as a proof of concept and for use with
    testing. Use at your own risk.

    Currently this context proxy supports the following CredSSP protocols:

        2:
            Earliest protocol that ships with Windows XP, Vista, 7, 8, Server
            2008, 2008 R2, and 2012. This version is susceptible to
            CVE-2018-0886
        3:
            Introduced with Windows 8.1 and Server 2012 R2. Adds the errorCode
            field for better error details on a failure. This version is
            susceptible to CVE-2018-0886.
        4:
            Largely the same as 3, introduced in an older Win 10 build.
        5:
            Same as version 2 but with the mitigations for CVE-2018-0886
            applied.
        6:
            Same as version 3 and 4 but with the mitigations for CVE-2018-0886
            applied.

    If the context proxy is talking to a peer with a higher protocol number it
    is treated the same as what the client supports.

    Optional kwargs supports by CredSSPProxy:

        credssp_negotiate_context: Use this contest for the underlying
            authentication negotiation. This allows the caller to restrict the
            auth to Kerberos or set any other setting specific to their
            environment:

        credssp_tls_context: Custom :class:`CredSSPTLSContext` to use for the
            CredSSP exchange. See `spnego.tls` for helper methods to generate
            a custom TLS context.

        credssp_min_protocol: The minimum CredSSP authentication protocol the
            context will allow. This can control whether the server rejects
            peers running on a lower protocol version than what is set to
            enforce a better security baseline. The default is 2 which works
            against all CredSSP peers. To ensure the mitigations for
            CVE-2018-0886 is in place set this value to 5 or higher.
    """

    def __init__(self, username: typing.Optional[typing.Union[str, Credential, typing.List[Credential]]]=None, password: typing.Optional[str]=None, hostname: typing.Optional[str]=None, service: typing.Optional[str]=None, channel_bindings: typing.Optional[GssChannelBindings]=None, context_req: ContextReq=ContextReq.default, usage: str='initiate', protocol: str='credssp', options: NegotiateOptions=NegotiateOptions.none, **kwargs: typing.Any) -> None:
        credentials = unify_credentials(username, password, required_protocol='credssp' if usage == 'initiate' else None)
        super(CredSSPProxy, self).__init__(credentials, hostname, service, channel_bindings, context_req, usage, protocol, options)
        if options & NegotiateOptions.session_key:
            raise FeatureMissingError(NegotiateOptions.session_key)
        self._credentials = credentials
        self._options = options & ~NegotiateOptions.wrapping_winrm
        self._auth_context: typing.Optional[ContextProxy] = kwargs.get('credssp_negotiate_context', None)
        self._ts_credential: typing.Optional[TSCredentials] = None
        self._complete = False
        self._step_gen: typing.Optional[typing.Generator[bytes, typing.Optional[bytes], None]] = None
        self._auth_stage = 'TLS Handshake'
        self._min_version: int = kwargs.get('credssp_min_protocol', 2)
        self._selected_version: typing.Optional[int] = None
        self._tls_context: CredSSPTLSContext
        if 'credssp_tls_context' in kwargs:
            self._tls_context = kwargs['credssp_tls_context']
            if usage == 'accept' and (not self._tls_context.public_key):
                raise OperationNotAvailableError(context_msg='Provided tls context does not have a public key set')
        else:
            self._tls_context = _create_tls_context(usage)
        self._in_buff = ssl.MemoryBIO()
        self._out_buff = ssl.MemoryBIO()
        self._tls_object = self._tls_context.context.wrap_bio(self._in_buff, self._out_buff, server_side=usage == 'accept')
        if usage == 'initiate':
            for cred in credentials:
                if isinstance(cred, Password):
                    domain, username = split_username(cred.username)
                    password_cred = TSPasswordCreds(domain or '', username or '', cred.password)
                    self._ts_credential = TSCredentials(credentials=password_cred)
                    break

    @classmethod
    def available_protocols(cls, options: typing.Optional[NegotiateOptions]=None) -> typing.List[str]:
        return ['credssp']

    @classmethod
    def iov_available(cls) -> bool:
        return False

    @property
    def client_principal(self) -> typing.Optional[str]:
        return self._auth_context.client_principal if self._auth_context else None

    @property
    def complete(self) -> bool:
        return self._complete

    @property
    def negotiated_protocol(self) -> typing.Optional[str]:
        return self._auth_context.negotiated_protocol if self._auth_context else None

    @property
    def session_key(self) -> bytes:
        raise OperationNotAvailableError(context_msg='CredSSP does not have a session key to share')

    def new_context(self) -> 'CredSSPProxy':
        return CredSSPProxy(username=self._credentials, hostname=self._hostname, service=self._service, channel_bindings=self.channel_bindings, context_req=self.context_req, usage=self.usage, protocol=self.protocol, options=self.options, credssp_negotiate_context=self._auth_context.new_context() if self._auth_context else None)

    def step(self, in_token: typing.Optional[bytes]=None, *, channel_bindings: typing.Optional[GssChannelBindings]=None) -> typing.Optional[bytes]:
        log.debug('CredSSP step input: %s', to_text(base64.b64encode(in_token or b'')))
        if self._step_gen is None:
            self._step_gen = getattr(self, '_step_%s' % self.usage)(in_token)
            in_token = None
        out_token = None
        try:
            out_token = self._step_gen.send(in_token)
        except StopIteration:
            pass
        log.debug('CredSSP step output: %s', to_text(base64.b64encode(out_token or b'')))
        return out_token

    def _step_initiate(self, in_token: typing.Optional[bytes]) -> typing.Generator[bytes, bytes, None]:
        """The initiator authentication steps of CredSSP."""
        yield from self._step_tls(in_token)
        server_certificate = typing.cast(bytes, self._tls_object.getpeercert(True))
        public_key = get_certificate_public_key(server_certificate)
        log.debug('Starting CredSSP authentication phase')
        if not self._auth_context:
            auth_kwargs: typing.Dict[str, typing.Any] = {}
            if self._hostname:
                auth_kwargs['hostname'] = self._hostname
            if self._service:
                auth_kwargs['service'] = self._service
            self._auth_context = spnego.client(self._credentials, protocol='negotiate', options=self._options, **auth_kwargs)
        round = 0
        out_token = self._auth_context.step()
        while True:
            round += 1
            auth_request = TSRequest(_CREDSSP_VERSION, nego_tokens=NegoData(out_token or b''))
            auth_response = (yield from self._yield_ts_request(auth_request, 'Authentication %d' % round))
            out_token = self._auth_context.step(auth_response.nego_tokens[0].nego_token)
            if self._auth_context.complete or (out_token and b'NTLMSSP\x00\x03\x00\x00\x00' in out_token):
                break
        self._selected_version = min(auth_response.version, _CREDSSP_VERSION)
        log.debug('Negotiated CredSSP version: %d' % self._selected_version)
        if self._selected_version < self._min_version:
            msg = f'The peer protocol version was {auth_response.version} and did not meet the minimum requirements of {self._min_version}'
            raise InvalidTokenError(context_msg=msg)
        pub_key_nego_token = NegoData(out_token) if out_token else None
        nonce = os.urandom(32) if self._selected_version > 4 else None
        pub_value = _get_pub_key_auth(public_key, 'initiate', nonce=nonce)
        pub_key_request = TSRequest(version=_CREDSSP_VERSION, nego_tokens=pub_key_nego_token, client_nonce=nonce, pub_key_auth=self._auth_context.wrap(pub_value).data)
        pub_key_response = (yield from self._yield_ts_request(pub_key_request, 'Public key exchange'))
        if not pub_key_response.pub_key_auth:
            raise InvalidTokenError(context_msg='Acceptor did not response with pubKeyAuth info.')
        if pub_key_response.nego_tokens:
            self._auth_context.step(pub_key_response.nego_tokens[0].nego_token)
        response_key = self._auth_context.unwrap(pub_key_response.pub_key_auth).data
        expected_key = _get_pub_key_auth(public_key, 'accept', nonce=nonce)
        if expected_key != response_key:
            raise BadBindingsError(context_msg='Public key verification failed, potential man in the middle attack')
        enc_credentials = self._auth_context.wrap(self._ts_credential.pack()).data
        credential_request = TSRequest(_CREDSSP_VERSION, auth_info=enc_credentials)
        self._complete = True
        yield from self._yield_ts_request(credential_request, 'Credential exchange')

    def _step_accept(self, in_token: typing.Optional[bytes]) -> typing.Generator[bytes, bytes, None]:
        """The acceptor authentication steps of CredSSP."""
        in_token = (yield from self._step_tls(in_token))
        auth_request = TSRequest.unpack(in_token or b'')
        self._selected_version = min(auth_request.version, _CREDSSP_VERSION)
        log.debug('Negotiated CredSSP version: %d' % self._selected_version)
        if self._selected_version < self._min_version:
            msg = f'The peer protocol version was {auth_request.version} and did not meet the minimum requirements of {self._min_version}'
            raise InvalidTokenError(context_msg=msg)
        try:
            log.debug('Starting CredSSP authentication phase')
            if not self._auth_context:
                auth_kwargs: typing.Dict[str, typing.Any] = {}
                if self._hostname:
                    auth_kwargs['hostname'] = self._hostname
                if self._service:
                    auth_kwargs['service'] = self._service
                self._auth_context = spnego.server(protocol='negotiate', options=self._options, **auth_kwargs)
            round = 0
            while True:
                round += 1
                nego_out_token = None
                if auth_request.nego_tokens:
                    nego_out_token = self._auth_context.step(auth_request.nego_tokens[0].nego_token)
                if auth_request.pub_key_auth:
                    break
                auth_response = TSRequest(_CREDSSP_VERSION, nego_tokens=NegoData(nego_out_token or b''))
                auth_request = (yield from self._yield_ts_request(auth_response, 'Authentication %d' % round))
        except SpnegoError as e:
            log.warning('Received CredSSP error when accepting credentials: %s', e)
            if self._selected_version in [3, 4] or self._selected_version >= 6:
                error_token = TSRequest(_CREDSSP_VERSION, error_code=e.nt_status)
                yield from self._yield_ts_request(error_token, 'Authentication failure')
            return
        actual_key = self._auth_context.unwrap(auth_request.pub_key_auth).data
        public_key = self._tls_context.public_key or b''
        expected_key = _get_pub_key_auth(public_key, 'initiate', nonce=auth_request.client_nonce)
        if actual_key != expected_key:
            raise BadBindingsError(context_msg='Public key verification failed, potential man in the middle attack')
        nego_token = NegoData(nego_out_token) if nego_out_token else None
        server_key = self._auth_context.wrap(_get_pub_key_auth(public_key, 'accept', nonce=auth_request.client_nonce)).data
        pub_key_response = TSRequest(_CREDSSP_VERSION, nego_tokens=nego_token, pub_key_auth=server_key)
        auth_request = (yield from self._yield_ts_request(pub_key_response, 'Public key exchange'))
        self._auth_stage = 'Credential exchange'
        if not auth_request.auth_info:
            raise InvalidTokenError(context_msg='No credential received on CredSSP TSRequest from initiator')
        self._ts_credential = TSCredentials.unpack(self._auth_context.unwrap(auth_request.auth_info).data)
        self._complete = True

    def _step_tls(self, in_token: typing.Optional[bytes]) -> typing.Generator[bytes, bytes, bytes]:
        """The TLS handshake phase of CredSSP."""
        try:
            while True:
                if in_token:
                    self._in_buff.write(in_token)
                want_read = False
                try:
                    self._tls_object.do_handshake()
                except ssl.SSLWantReadError:
                    want_read = True
                out_token = self._out_buff.read()
                if not out_token:
                    break
                in_token = (yield out_token)
                if not want_read and self.usage == 'accept':
                    out_token = self.unwrap(in_token).data
                    break
        except ssl.SSLError as e:
            raise InvalidTokenError(context_msg='TLS handshake for CredSSP: %s' % e) from e
        cipher, protocol, _ = self._tls_object.cipher()
        log.debug('TLS handshake complete, negotiation details: %s %s', protocol, cipher)
        return out_token

    def _yield_ts_request(self, ts_request: TSRequest, context_msg: str) -> typing.Generator[bytes, bytes, TSRequest]:
        """Exchanges a TSRequest between the initiator and acceptor."""
        self._auth_stage = context_msg
        out_request = ts_request.pack()
        log.debug('CredSSP TSRequest output: %s' % to_text(base64.b64encode(out_request)))
        wrapped_response = (yield self.wrap(out_request).data)
        in_request = self.unwrap(wrapped_response).data
        log.debug('CredSSP TSRequest input: %s' % to_text(base64.b64encode(in_request)))
        response = TSRequest.unpack(in_request)
        if response.error_code:
            base_error = NativeError('Received NTStatus in TSRequest from acceptor', winerror=response.error_code)
            raise SpnegoError(base_error=base_error, context_msg=context_msg)
        return response

    def get_extra_info(self, name: str, default: typing.Any=None) -> typing.Any:
        if name == 'client_credential' and self.usage == 'accept' and self._ts_credential:
            return self._ts_credential.credentials
        elif name == 'sslcontext':
            return self._tls_context.context
        elif name == 'ssl_object':
            return self._tls_object
        elif name == 'auth_stage':
            return self._auth_stage
        elif name == 'protocol_version' and self._selected_version is not None:
            return self._selected_version
        else:
            return default

    def query_message_sizes(self) -> SecPkgContextSizes:
        if not self._tls_object or not self.complete:
            raise NoContextError(context_msg='Cannot get message sizes until context has been established')
        cipher_negotiated, tls_protocol, _ = self._tls_object.cipher()
        trailer_length = _tls_trailer_length(0, tls_protocol, cipher_negotiated)
        return SecPkgContextSizes(header=trailer_length)

    @_wrap_ssl_error('Invalid TLS state when wrapping data')
    def wrap(self, data: bytes, encrypt: bool=True, qop: typing.Optional[int]=None) -> WrapResult:
        self._tls_object.write(data)
        return WrapResult(data=self._out_buff.read(), encrypted=True)

    def wrap_iov(self, iov: typing.Iterable[IOV], encrypt: bool=True, qop: typing.Optional[int]=None) -> IOVWrapResult:
        raise OperationNotAvailableError(context_msg='CredSSP does not offer IOV wrapping')

    def wrap_winrm(self, data: bytes) -> WinRMWrapResult:
        enc_data = self.wrap(data).data
        cipher_negotiated, tls_protocol, _ = self._tls_object.cipher()
        trailer_length = _tls_trailer_length(len(data), tls_protocol, cipher_negotiated)
        return WinRMWrapResult(header=enc_data[:trailer_length], data=enc_data[trailer_length:], padding_length=0)

    @_wrap_ssl_error('Invalid TLS state when unwrapping data')
    def unwrap(self, data: bytes) -> UnwrapResult:
        self._in_buff.write(data)
        chunks = []
        while True:
            try:
                chunks.append(self._tls_object.read())
            except ssl.SSLWantReadError:
                break
        return UnwrapResult(data=b''.join(chunks), encrypted=True, qop=0)

    def unwrap_iov(self, iov: typing.Iterable[IOV]) -> IOVUnwrapResult:
        raise OperationNotAvailableError(context_msg='CredSSP does not offer IOV wrapping')

    def unwrap_winrm(self, header: bytes, data: bytes) -> bytes:
        return self.unwrap(header + data).data

    def sign(self, data: bytes, qop: typing.Optional[int]=None) -> bytes:
        raise OperationNotAvailableError(context_msg='CredSSP does not offer signing')

    def verify(self, data: bytes, mic: bytes) -> int:
        raise OperationNotAvailableError(context_msg='CredSSP does not offer verification')

    @property
    def _context_attr_map(self) -> typing.List[typing.Tuple[ContextReq, int]]:
        return []