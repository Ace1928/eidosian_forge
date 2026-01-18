from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.core import exceptions
def _RecordConditionComplete(self, condition):
    """Take care of the internal-to-this-class bookkeeping stage complete."""
    for requirements in self._dependencies.values():
        requirements.discard(condition)