from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.core import exceptions
def _PotentiallyUpdateInstanceCompletions(self, job_obj, conditions):
    """Maybe update the terminal condition stage message with number of completions."""
    terminal_condition = conditions.TerminalCondition()
    if terminal_condition not in self._tracker or self._IsBlocked(terminal_condition):
        return
    self._tracker.UpdateStage(terminal_condition, '{} / {} complete'.format(job_obj.status.succeededCount or 0, job_obj.task_count))