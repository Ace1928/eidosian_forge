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
def CancelOperationPrompt(op_name):
    """Prompt the user before cancelling an LRO operation.

  Args:
    op_name: str, name of the operation.
  """
    message = 'The operation {0} will be cancelled.'
    console_io.PromptContinue(message=message.format(op_name), throw_if_unattended=True, cancel_on_no=True)