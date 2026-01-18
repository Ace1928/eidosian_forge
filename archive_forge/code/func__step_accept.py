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
def _step_accept(self, in_token: bytes, *, channel_bindings: typing.Optional[GssChannelBindings]=None) -> typing.Optional[bytes]:
    if not self._temp_negotiate:
        return self._step_accept_negotiate(in_token)
    else:
        self._step_accept_authenticate(in_token, channel_bindings=channel_bindings or self.channel_bindings)
        return None