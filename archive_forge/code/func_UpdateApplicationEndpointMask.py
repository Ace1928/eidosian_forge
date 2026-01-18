from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.beyondcorp.app import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.beyondcorp.app import util as command_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def UpdateApplicationEndpointMask(unused_ref, args, patch_request):
    """Updates application-endpoint mask."""
    if args.IsSpecified('application_endpoint'):
        patch_request = command_util.AddFieldToUpdateMask('application_endpoint', patch_request)
    return patch_request