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