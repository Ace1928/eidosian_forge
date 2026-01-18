from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import util as gke_util
from googlecloudsdk.api_lib.container.gkemulticloud import operations as op_api_util
from googlecloudsdk.api_lib.container.gkemulticloud import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.run import pretty_print
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def CancelOperationMessage(name, kind):
    """Message to display after cancelling an LRO operation.

  Args:
    name: str, name of the operation.
    kind: str, the kind of LRO operation e.g. AWS or Azure.

  Returns:
    The operation cancellation message.
  """
    msg = 'Cancelation of operation {0} has been requested. Please use gcloud container {1} operations describe {2} to check if the operation has been cancelled successfully.'
    return msg.format(name, kind, name)