from ... import lazy_import
from breezy.bzr.smart import request as _mod_request
import breezy
from ... import debug, errors, hooks, trace
from . import message, protocol
def _send_no_retry(self, encoder):
    """Just encode the request and try to send it."""
    encoder.set_headers(self.client._headers)
    if self.body is not None:
        if self.readv_body is not None:
            raise AssertionError('body and readv_body are mutually exclusive.')
        if self.body_stream is not None:
            raise AssertionError('body and body_stream are mutually exclusive.')
        encoder.call_with_body_bytes((self.method,) + self.args, self.body)
    elif self.readv_body is not None:
        if self.body_stream is not None:
            raise AssertionError('readv_body and body_stream are mutually exclusive.')
        encoder.call_with_body_readv_array((self.method,) + self.args, self.readv_body)
    elif self.body_stream is not None:
        encoder.call_with_body_stream((self.method,) + self.args, self.body_stream)
    else:
        encoder.call(self.method, *self.args)