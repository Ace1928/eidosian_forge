from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import exceptions as sdk_core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
import six
def SetSpot(api_version):
    """Creates an empty Spot structure if spot flag is set."""

    def Process(ref, args, request):
        del ref
        if args.spot:
            tpu_messages = GetMessagesModule(api_version)
            if request.queuedResource is None:
                request.queuedResource = tpu_messages.QueuedResource()
            if request.queuedResource.spot is None:
                request.queuedResource.spot = tpu_messages.Spot()
        return request
    return Process