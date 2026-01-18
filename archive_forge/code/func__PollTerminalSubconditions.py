from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.core import exceptions
def _PollTerminalSubconditions(self, conditions, conditions_message):
    for condition in conditions.TerminalSubconditions():
        if condition not in self._dependencies:
            continue
        message = conditions[condition]['message']
        status = conditions[condition]['status']
        self._PossiblyUpdateMessage(condition, message, conditions_message)
        if status is None:
            continue
        elif status:
            if self._PossiblyCompleteStage(condition, message):
                self._PollTerminalSubconditions(conditions, conditions_message)
                break
        else:
            self._PossiblyFailStage(condition, message)