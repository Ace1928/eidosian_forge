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