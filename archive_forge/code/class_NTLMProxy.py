import base64
import logging
import os
import socket
import typing
from spnego._context import (
from spnego._credential import (
from spnego._ntlm_raw.crypto import (
from spnego._ntlm_raw.messages import (
from spnego._ntlm_raw.security import seal, sign
from spnego._text import to_text
from spnego.channel_bindings import GssChannelBindings
from spnego.exceptions import (
from spnego.iov import BufferType, IOVResBuffer
class NTLMProxy(ContextProxy):
    """A context wrapper for a Python managed NTLM context.

    This is a context that can be used on Linux to generate NTLM without any system dependencies.
    """

    def __init__(self, username: typing.Optional[typing.Union[str, Credential, typing.List[Credential]]]=None, password: typing.Optional[str]=None, hostname: typing.Optional[str]=None, service: typing.Optional[str]=None, channel_bindings: typing.Optional[GssChannelBindings]=None, context_req: ContextReq=ContextReq.default, usage: str='initiate', protocol: str='ntlm', options: NegotiateOptions=NegotiateOptions.none, **kwargs: typing.Any) -> None:
        credentials = unify_credentials(username, password, required_protocol='ntlm')
        super(NTLMProxy, self).__init__(credentials, hostname, service, channel_bindings, context_req, usage, protocol, options)
        self._complete = False
        self._credential: typing.Optional[_NTLMCredential] = None
        self._context_req = self._context_req | NegotiateFlags.key_128 | NegotiateFlags.key_56 | NegotiateFlags.key_exch | NegotiateFlags.extended_session_security | NegotiateFlags.always_sign | NegotiateFlags.ntlm | NegotiateFlags.lm_key | NegotiateFlags.request_target | NegotiateFlags.oem | NegotiateFlags.unicode
        lm_compat_level = int(os.environ.get('LM_COMPAT_LEVEL', 3))
        if lm_compat_level < 0 or lm_compat_level > 5:
            raise SpnegoError(ErrorCode.failure, context_msg='Invalid LM_COMPAT_LEVEL %d, must be between 0 and 5' % lm_compat_level)
        if lm_compat_level == 0:
            self._context_req &= ~NegotiateFlags.extended_session_security
        if self.usage == 'initiate':
            self._credential = _NTLMCredential(next((c for c in credentials if 'ntlm' in c.supported_protocols)))
            self._lm = lm_compat_level < 2
            self._nt_v1 = lm_compat_level < 3
            self._nt_v2 = lm_compat_level > 2
            if lm_compat_level > 1:
                self._context_req &= ~NegotiateFlags.lm_key
        else:
            self._lm = lm_compat_level < 4
            self._nt_v1 = lm_compat_level < 5
            self._nt_v2 = True
            if not _get_credential_file():
                raise OperationNotAvailableError(context_msg='Retrieving NTLM store without NTLM_USER_FILE set to a filepath')
        self._temp_negotiate: typing.Optional[Negotiate] = None
        self._temp_challenge: typing.Optional[Challenge] = None
        self._mic_required = False
        self._session_key: typing.Optional[bytes] = None
        self._sign_key_out: typing.Optional[bytes] = None
        self._sign_key_in: typing.Optional[bytes] = None
        self._handle_out: typing.Optional[RC4Handle] = None
        self._handle_in: typing.Optional[RC4Handle] = None
        self.__seq_num_in = 0
        self.__seq_num_out = 0

    @classmethod
    def available_protocols(cls, options: typing.Optional[NegotiateOptions]=None) -> typing.List[str]:
        return ['ntlm']

    @classmethod
    def iov_available(cls) -> bool:
        return True

    @property
    def client_principal(self) -> typing.Optional[str]:
        if self.usage == 'accept' and self.complete and self._credential:
            domain_part = self._credential.domain + '\\' if self._credential.domain else ''
            return '%s%s' % (domain_part, self._credential.username)
        return None

    @property
    def complete(self) -> bool:
        return self._complete

    @property
    def negotiated_protocol(self) -> typing.Optional[str]:
        return 'ntlm'

    @property
    def session_key(self) -> bytes:
        return self._session_key or b''

    def new_context(self) -> 'NTLMProxy':
        cred: typing.Optional[NTLMHash] = None
        if self._credential and self.usage == 'initiate':
            cred = NTLMHash(username=self._credential._raw_username or '', lm_hash=base64.b16encode(self._credential.lm_hash).decode(), nt_hash=base64.b16encode(self._credential.nt_hash).decode())
        return NTLMProxy(username=cred, hostname=self._hostname, service=self._service, channel_bindings=self.channel_bindings, context_req=self.context_req, usage=self.usage, protocol=self.protocol, options=self.options)

    def step(self, in_token: typing.Optional[bytes]=None, *, channel_bindings: typing.Optional[GssChannelBindings]=None) -> typing.Optional[bytes]:
        if not self._is_wrapped:
            log.debug('NTLM step input: %s', base64.b64encode(in_token or b'').decode())
        out_token = getattr(self, '_step_%s' % self.usage)(in_token=in_token, channel_bindings=channel_bindings)
        if not self._is_wrapped:
            log.debug('NTLM step output: %s', base64.b64encode(out_token or b'').decode())
        if self._complete:
            self._temp_negotiate = None
            self._temp_challenge = None
            in_usage = 'accept' if self.usage == 'initiate' else 'initiate'
            session_key = self._session_key or b''
            self._sign_key_out = signkey(self._context_attr, session_key, self.usage)
            self._sign_key_in = signkey(self._context_attr, session_key, in_usage)
            if self._context_attr & NegotiateFlags.extended_session_security:
                self._handle_out = rc4init(sealkey(self._context_attr, session_key, self.usage))
                self._handle_in = rc4init(sealkey(self._context_attr, session_key, in_usage))
            else:
                self._handle_out = self._handle_in = rc4init(sealkey(self._context_attr, session_key, self.usage))
        return out_token

    def _step_initiate(self, in_token: typing.Optional[bytes]=None, *, channel_bindings: typing.Optional[GssChannelBindings]=None) -> bytes:
        if not self._temp_negotiate:
            self._temp_negotiate = Negotiate(self._context_req, version=Version.get_current())
            return self._temp_negotiate.pack()
        in_token = in_token or b''
        challenge = Challenge.unpack(in_token)
        credential = typing.cast(_NTLMCredential, self._credential)
        auth_kwargs: typing.Dict[str, typing.Any] = {'domain_name': credential.domain, 'username': credential.username}
        if challenge.flags & NegotiateFlags.version:
            auth_kwargs['version'] = Version.get_current()
            auth_kwargs['workstation'] = _get_workstation()
        nt_challenge, lm_challenge, key_exchange_key = self._compute_response(challenge, credential, channel_bindings=channel_bindings or self.channel_bindings)
        if challenge.flags & NegotiateFlags.key_exch:
            if challenge.flags & NegotiateFlags.sign or challenge.flags & NegotiateFlags.seal:
                self._session_key = os.urandom(16)
                auth_kwargs['encrypted_session_key'] = rc4k(key_exchange_key, self._session_key)
            else:
                self._session_key = key_exchange_key
                auth_kwargs['encrypted_session_key'] = b'\x00'
        else:
            self._session_key = key_exchange_key
        authenticate = Authenticate(challenge.flags, lm_challenge, nt_challenge, **auth_kwargs)
        if self._mic_required:
            authenticate.mic = self._calculate_mic(self._session_key, self._temp_negotiate.pack(), in_token, authenticate.pack())
        self._context_attr = authenticate.flags
        self._complete = True
        return authenticate.pack()

    def _step_accept(self, in_token: bytes, *, channel_bindings: typing.Optional[GssChannelBindings]=None) -> typing.Optional[bytes]:
        if not self._temp_negotiate:
            return self._step_accept_negotiate(in_token)
        else:
            self._step_accept_authenticate(in_token, channel_bindings=channel_bindings or self.channel_bindings)
            return None

    def _step_accept_negotiate(self, token: bytes) -> bytes:
        """Process the Negotiate message from the initiator."""
        negotiate = Negotiate.unpack(token)
        flags = negotiate.flags | NegotiateFlags.request_target | NegotiateFlags.ntlm | NegotiateFlags.always_sign | NegotiateFlags.target_info | NegotiateFlags.target_type_server
        if flags & NegotiateFlags.unicode:
            flags &= ~NegotiateFlags.oem
        elif flags & NegotiateFlags.oem == 0:
            raise SpnegoError(ErrorCode.failure, context_msg='Neither NEGOTIATE_OEM or NEGOTIATE_UNICODE flags were set, cannot derive encoding for text fields')
        if flags & NegotiateFlags.extended_session_security:
            flags &= ~NegotiateFlags.lm_key
        server_challenge = os.urandom(8)
        target_name = to_text(socket.gethostname()).upper()
        target_info = TargetInfo()
        target_info[AvId.nb_computer_name] = target_name
        target_info[AvId.nb_domain_name] = 'WORKSTATION'
        target_info[AvId.dns_computer_name] = to_text(socket.getfqdn())
        target_info[AvId.timestamp] = FileTime.now()
        challenge = Challenge(flags, server_challenge, target_name=target_name, target_info=target_info)
        self._temp_negotiate = negotiate
        self._temp_challenge = challenge
        return challenge.pack()

    def _step_accept_authenticate(self, token: bytes, channel_bindings: typing.Optional[GssChannelBindings]) -> None:
        """Process the Authenticate message from the initiator."""
        negotiate = typing.cast(Negotiate, self._temp_negotiate)
        challenge = typing.cast(Challenge, self._temp_challenge)
        server_challenge = challenge.server_challenge
        auth = Authenticate.unpack(token)
        if not auth.user_name or (not auth.nt_challenge_response and (not auth.lm_challenge_response or auth.lm_challenge_response == b'\x00')):
            raise OperationNotAvailableError(context_msg='Anonymous user authentication not implemented')
        username = auth.user_name
        if auth.domain_name:
            username = f'{auth.domain_name}\\{username}'
        self._credential = _NTLMCredential(CredentialCache(username=username))
        expected_mic = None
        if auth.nt_challenge_response and len(auth.nt_challenge_response) > 24:
            nt_hash = ntowfv2(auth.user_name, self._credential.nt_hash, auth.domain_name)
            nt_challenge = NTClientChallengeV2.unpack(auth.nt_challenge_response[16:])
            time = nt_challenge.time_stamp
            client_challenge = nt_challenge.challenge_from_client
            target_info = nt_challenge.av_pairs
            expected_nt, expected_lm, key_exchange_key = compute_response_v2(nt_hash, server_challenge, client_challenge, time, target_info)
            if channel_bindings:
                if AvId.channel_bindings not in target_info:
                    raise BadBindingsError(context_msg='Acceptor bindings specified but not present in initiator response')
                expected_bindings = target_info[AvId.channel_bindings]
                actual_bindings = md5(channel_bindings.pack())
                if expected_bindings not in [actual_bindings, b'\x00' * 16]:
                    raise BadBindingsError(context_msg='Acceptor bindings do not match initiator bindings')
            if target_info.get(AvId.flags, 0) & AvFlags.mic:
                expected_mic = auth.mic
        else:
            if not self._nt_v1:
                raise InvalidTokenError(context_msg='Acceptor settings are set to reject NTv1 responses')
            elif not auth.nt_challenge_response and (not self._lm):
                raise InvalidTokenError(context_msg='Acceptor settings are set to reject LM responses')
            client_challenge = b'\x00' * 8
            if auth.flags & NegotiateFlags.extended_session_security:
                client_challenge = (auth.lm_challenge_response or b'\x00' * 8)[:8]
            expected_nt, expected_lm, key_exchange_key = compute_response_v1(auth.flags, self._credential.nt_hash, self._credential.lm_hash, server_challenge, client_challenge, no_lm_response=not self._lm)
        auth_success = False
        if auth.nt_challenge_response:
            auth_success = auth.nt_challenge_response == expected_nt
        elif auth.lm_challenge_response:
            auth_success = auth.lm_challenge_response == expected_lm
        if not auth_success:
            raise InvalidTokenError(context_msg='Invalid NTLM response from initiator')
        if auth.flags & NegotiateFlags.key_exch and (auth.flags & NegotiateFlags.sign or auth.flags & NegotiateFlags.seal):
            self._session_key = rc4k(key_exchange_key, auth.encrypted_random_session_key or b'')
        else:
            self._session_key = key_exchange_key
        if expected_mic:
            auth.mic = b'\x00' * 16
            actual_mic = self._calculate_mic(self.session_key, negotiate.pack(), challenge.pack(), auth.pack())
            if actual_mic != expected_mic:
                raise InvalidTokenError(context_msg='Invalid MIC in NTLM authentication message')
        self._context_attr = auth.flags
        self._complete = True

    def query_message_sizes(self) -> SecPkgContextSizes:
        if not self.complete:
            raise NoContextError(context_msg='Cannot get message sizes until context has been established')
        return SecPkgContextSizes(header=16)

    def wrap(self, data: bytes, encrypt: bool=True, qop: typing.Optional[int]=None) -> WrapResult:
        if qop:
            raise UnsupportedQop(context_msg='Unsupported QoP value %s specified for NTLM' % qop)
        if self.context_attr & ContextReq.integrity == 0 and self.context_attr & ContextReq.confidentiality == 0:
            raise OperationNotAvailableError(context_msg='NTLM wrap without integrity or confidentiality')
        if not self._handle_out or self._sign_key_out is None:
            raise NoContextError(context_msg='Cannot wrap until context has been established')
        msg, signature = seal(self._context_attr, self._handle_out, self._sign_key_out, self._seq_num_out, data)
        return WrapResult(data=signature + msg, encrypted=True)

    def wrap_iov(self, iov: typing.Iterable[IOV], encrypt: bool=True, qop: typing.Optional[int]=None) -> IOVWrapResult:
        if qop:
            raise UnsupportedQop(context_msg='Unsupported QoP value %s specified for NTLM' % qop)
        if self.context_attr & ContextReq.integrity == 0 and self.context_attr & ContextReq.confidentiality == 0:
            raise OperationNotAvailableError(context_msg='NTLM wrap without integrity or confidentiality')
        if not self._handle_out or self._sign_key_out is None:
            raise NoContextError(context_msg='Cannot wrap until context has been established')
        header_idx = -1
        data_idx = -1
        signature_input = []
        buffers = self._build_iov_list(iov, lambda b: b)
        res: typing.List[IOVResBuffer] = []
        for idx, buffer in enumerate(buffers):
            data: bytes
            if buffer.type == BufferType.header:
                if header_idx != -1:
                    raise InvalidTokenError(context_msg='wrap_iov must only be used with 1 header IOV buffer.')
                header_idx = idx
                data = b''
            elif buffer.type == BufferType.data:
                if data_idx != -1:
                    raise InvalidTokenError(context_msg='wrap_iov must only be used with 1 data IOV buffer.')
                if not isinstance(buffer.data, bytes):
                    raise InvalidTokenError(context_msg=f'wrap_iov IOV data buffer at [{idx}] must be bytes')
                data_idx = idx
                data = buffer.data
                signature_input.append(data)
            elif buffer.type in [BufferType.sign_only, BufferType.data_readonly]:
                if not isinstance(buffer.data, bytes):
                    raise InvalidTokenError(context_msg=f'wrap_iov IOV {buffer.type.name} buffer at [{idx}] must be bytes')
                data = buffer.data
                signature_input.append(data)
            else:
                raise InvalidTokenError(context_msg=f'wrap_iov unsupported IOV buffer type {buffer.type.name}')
            res.append(IOVResBuffer(buffer.type, data))
        if header_idx == -1:
            raise InvalidTokenError(context_msg='wrap_iov no IOV header buffer present')
        if data_idx == -1:
            raise InvalidTokenError(context_msg='wrap_iov no IOV data buffer present')
        enc_msg, signature = seal(self._context_attr, self._handle_out, self._sign_key_out, self._seq_num_out, res[data_idx][1] or b'', to_sign=b''.join(signature_input))
        res[header_idx] = IOVResBuffer(BufferType.header, signature)
        res[data_idx] = IOVResBuffer(BufferType.data, enc_msg)
        return IOVWrapResult(tuple(res), encrypted=True)

    def wrap_winrm(self, data: bytes) -> WinRMWrapResult:
        enc_data = self.wrap(data).data
        return WinRMWrapResult(header=enc_data[:16], data=enc_data[16:], padding_length=0)

    def unwrap(self, data: bytes) -> UnwrapResult:
        if not self._handle_in:
            raise NoContextError(context_msg='Cannot unwrap until context has been established')
        signature = data[:16]
        msg = self._handle_in.update(data[16:])
        self.verify(msg, signature)
        return UnwrapResult(data=msg, encrypted=True, qop=0)

    def unwrap_iov(self, iov: typing.Iterable[IOV]) -> IOVUnwrapResult:
        if self.context_attr & ContextReq.integrity == 0 and self.context_attr & ContextReq.confidentiality == 0:
            raise OperationNotAvailableError(context_msg='NTLM unwrap without integrity or confidentiality')
        if not self._handle_in or self._sign_key_in is None:
            raise NoContextError(context_msg='Cannot unwrap until context has been established')
        buffers = self._build_iov_list(iov, lambda b: b)
        if len(buffers) == 2 and buffers[0].type == BufferType.stream and (buffers[1].type == BufferType.data) and isinstance(buffers[0].data, bytes):
            unwrap_res = self.unwrap(buffers[0].data)
            return IOVUnwrapResult((IOVResBuffer(BufferType.stream, buffers[0].data), IOVResBuffer(BufferType.data, unwrap_res.data)), encrypted=unwrap_res.encrypted, qop=unwrap_res.qop)
        header_idx = -1
        data_idx = -1
        data_sig_idx = -1
        signature_input: typing.List[bytes] = []
        res: typing.List[IOVResBuffer] = []
        for idx, buffer in enumerate(buffers):
            data: bytes
            if not isinstance(buffer.data, bytes):
                raise InvalidTokenError(context_msg=f'unwrap_iov IOV {buffer.type.name} buffer at [{idx}] must be bytes')
            data = buffer.data
            if buffer.type == BufferType.header:
                if header_idx != -1:
                    raise InvalidTokenError(context_msg='unwrap_iov must only be used with 1 header IOV buffer.')
                header_idx = idx
            elif buffer.type == BufferType.data:
                if data_idx != -1:
                    raise InvalidTokenError(context_msg='unwrap_iov must only be used with 1 data IOV buffer.')
                data_idx = idx
                data_sig_idx = len(signature_input)
                signature_input.append(b'')
            elif buffer.type in [BufferType.sign_only, BufferType.data_readonly]:
                signature_input.append(data)
            else:
                raise InvalidTokenError(context_msg=f'unwrap_iov unsupported IOV buffer type {buffer.type.name}')
            res.append(IOVResBuffer(buffer.type, data))
        if header_idx == -1:
            raise InvalidTokenError(context_msg='unwrap_iov no IOV header buffer present')
        if data_idx == -1:
            raise InvalidTokenError(context_msg='unwrap_iov no IOV data buffer present')
        dec = self._handle_in.update(res[data_idx].data or b'')
        res[data_idx] = IOVResBuffer(BufferType.data, dec)
        signature_input[data_sig_idx] = dec
        self.verify(b''.join(signature_input), res[header_idx].data or b'')
        return IOVUnwrapResult(tuple(res), encrypted=True, qop=0)

    def unwrap_winrm(self, header: bytes, data: bytes) -> bytes:
        if not self._handle_in:
            raise NoContextError(context_msg='Cannot unwrap until context has been established')
        msg = self._handle_in.update(data)
        self.verify(msg, header)
        return msg

    def sign(self, data: bytes, qop: typing.Optional[int]=None) -> bytes:
        if qop:
            raise UnsupportedQop(context_msg='Unsupported QoP value %s specified for NTLM' % qop)
        if not self._handle_out or self._sign_key_out is None:
            raise NoContextError(context_msg='Cannot sign until context has been established')
        return sign(self._context_attr, self._handle_out, self._sign_key_out, self._seq_num_out, data)

    def verify(self, data: bytes, mic: bytes) -> int:
        if not self._handle_in or self._sign_key_in is None:
            raise NoContextError(context_msg='Cannot verify until context has been established')
        expected_sig = sign(self._context_attr, self._handle_in, self._sign_key_in, self._seq_num_in, data)
        if expected_sig != mic:
            raise BadMICError(context_msg='Invalid Message integrity Check (MIC) detected')
        return 0

    @property
    def _context_attr_map(self) -> typing.List[typing.Tuple[ContextReq, int]]:
        return [(ContextReq.replay_detect, NegotiateFlags.sign), (ContextReq.sequence_detect, NegotiateFlags.sign), (ContextReq.confidentiality, NegotiateFlags.seal), (ContextReq.integrity, NegotiateFlags.sign)]

    @property
    def _requires_mech_list_mic(self) -> bool:
        if not self._complete:
            self._mic_required = True
            return False
        return self._mic_required

    @property
    def _seq_num_in(self) -> int:
        if self._context_attr & NegotiateFlags.extended_session_security:
            num = self.__seq_num_in
            self.__seq_num_in += 1
        else:
            num = self.__seq_num_out
            self.__seq_num_out += 1
        return num

    @property
    def _seq_num_out(self) -> int:
        num = self.__seq_num_out
        self.__seq_num_out += 1
        return num

    def _calculate_mic(self, session_key: bytes, negotiate: bytes, challenge: bytes, authenticate: bytes) -> bytes:
        """Calculates the MIC value for the negotiated context."""
        return hmac_md5(session_key, negotiate + challenge + authenticate)

    def _compute_response(self, challenge: Challenge, credential: _NTLMCredential, channel_bindings: typing.Optional[GssChannelBindings]) -> typing.Tuple[bytes, bytes, bytes]:
        """Compute the NT and LM responses and the key exchange key."""
        client_challenge = os.urandom(8)
        if self._nt_v2:
            target_info = challenge.target_info.copy() if challenge.target_info else TargetInfo()
            if AvId.timestamp in target_info:
                time = target_info[AvId.timestamp]
                self._mic_required = True
            else:
                time = FileTime.now()
            if channel_bindings:
                target_info[AvId.channel_bindings] = md5(channel_bindings.pack())
            target_info[AvId.target_name] = self.spn or ''
            if self._mic_required:
                target_info[AvId.flags] = target_info.get(AvId.flags, AvFlags(0)) | AvFlags.mic
            ntv2_hash = ntowfv2(credential.username or '', credential.nt_hash, credential.domain)
            nt_challenge, lm_challenge, key_exchange_key = compute_response_v2(ntv2_hash, challenge.server_challenge, client_challenge, time, target_info)
            if self._mic_required:
                lm_challenge = b'\x00' * 24
            return (nt_challenge, lm_challenge, key_exchange_key)
        else:
            return compute_response_v1(challenge.flags, credential.nt_hash, credential.lm_hash, challenge.server_challenge, client_challenge, no_lm_response=not self._lm)

    def _reset_ntlm_crypto_state(self, outgoing: bool=True) -> None:
        direction = 'out' if outgoing else 'in'
        getattr(self, f'_handle_{direction}').reset()