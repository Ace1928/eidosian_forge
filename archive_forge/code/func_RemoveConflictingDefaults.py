from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def RemoveConflictingDefaults(unused_ref, args, request):
    """Unset acceleratorType flag when it conflicts with topology arguments.

  Args:
    unused_ref: ref to the service.
    args:  The args for this method.
    request: The request to be made.

  Returns:
    Request with metadata field populated.
  """
    if args.topology is not None:
        request.node.acceleratorType = None
    return request