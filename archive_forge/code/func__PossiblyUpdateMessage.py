from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.core import exceptions
def _PossiblyUpdateMessage(self, condition, message, conditions_message):
    """Update the stage message.

    Args:
      condition: str, The name of the status condition.
      message: str, The new message to display
      conditions_message: str, The message from the conditions object we're
        displaying..
    """
    if condition not in self._tracker or self._tracker.IsComplete(condition):
        return
    if self._IsBlocked(condition):
        return
    if message != conditions_message:
        self._tracker.UpdateStage(condition, message)