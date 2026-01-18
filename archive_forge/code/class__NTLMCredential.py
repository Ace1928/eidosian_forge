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
class _NTLMCredential:

    def __init__(self, credential: typing.Optional[Credential]=None) -> None:
        self._raw_username: typing.Optional[str] = None
        self._store: typing.Optional[str]
        if isinstance(credential, Password):
            self._store = 'explicit'
            self._raw_username = credential.username
            self.domain, self.username = split_username(credential.username)
            self.lm_hash = lmowfv1(credential.password)
            self.nt_hash = ntowfv1(credential.password)
        elif isinstance(credential, NTLMHash):
            self._store = 'explicit'
            self._raw_username = credential.username
            self.domain, self.username = split_username(credential.username)
            self.lm_hash = base64.b16decode(credential.lm_hash.upper()) if credential.lm_hash else b'\x00' * 16
            self.nt_hash = base64.b16decode(credential.nt_hash.upper()) if credential.nt_hash else b'\x00' * 16
        else:
            domain = username = None
            if isinstance(credential, CredentialCache):
                self._raw_username = credential.username
                domain, username = split_username(credential.username)
            self._store = _get_credential_file()
            self.domain, self.username, self.lm_hash, self.nt_hash = _get_credential(self._store, domain, username)