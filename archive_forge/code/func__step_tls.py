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