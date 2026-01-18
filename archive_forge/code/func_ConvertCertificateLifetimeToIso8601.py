from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.privateca import base
from googlecloudsdk.api_lib.privateca import request_utils
from googlecloudsdk.command_lib.privateca import resource_args
from googlecloudsdk.core.util import times
def ConvertCertificateLifetimeToIso8601(response, unused_args):
    """Converts certificate lifetimes from proto duration format to ISO8601."""
    if response.lifetime:
        response.lifetime = _ConvertProtoToIsoDuration(response.lifetime)
    if response.certificateDescription and response.certificateDescription.subjectDescription and response.certificateDescription.subjectDescription.lifetime:
        response.certificateDescription.subjectDescription.lifetime = _ConvertProtoToIsoDuration(response.certificateDescription.subjectDescription.lifetime)
    return response