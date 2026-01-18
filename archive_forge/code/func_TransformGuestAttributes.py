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
def TransformGuestAttributes(response, args):
    """Transforms the GuestAttributes into a flatter list.

  This is needed to make clearer output in the case of TPU pods, since they have
  many workers.

  Args:
    response: response to GetGuestAttributes.
    args: the arguments for the GetGuestAttributes command.

  Returns:
    A list of GuestAttributesListEntry objects.
  """
    del args
    lst = []
    for i, ga in enumerate(response.guestAttributes):
        for entry in ga.queryValue.items:
            lst.append(GuestAttributesListEntry(i, entry.namespace, entry.key, entry.value))
    return lst