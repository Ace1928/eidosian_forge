from ... import lazy_import
from breezy.bzr.smart import request as _mod_request
import breezy
from ... import debug, errors, hooks, trace
from . import message, protocol
def _construct_protocol(self, version):
    """Build the encoding stack for a given protocol version."""
    request = self.client._medium.get_request()
    if version == 3:
        request_encoder = protocol.ProtocolThreeRequester(request)
        response_handler = message.ConventionalResponseHandler()
        response_proto = protocol.ProtocolThreeDecoder(response_handler, expect_version_marker=True)
        response_handler.setProtoAndMediumRequest(response_proto, request)
    elif version == 2:
        request_encoder = protocol.SmartClientRequestProtocolTwo(request)
        response_handler = request_encoder
    else:
        request_encoder = protocol.SmartClientRequestProtocolOne(request)
        response_handler = request_encoder
    return (request_encoder, response_handler)