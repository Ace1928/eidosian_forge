from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.beyondcorp.app import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.beyondcorp.app import util as command_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
def HideDetailsBeforeDescribing(response, args):
    """Hide details before describing a connector."""
    if args.details or response.resourceInfo is None:
        return response
    response.resourceInfo.resource = None
    response.resourceInfo.sub.clear()
    return response