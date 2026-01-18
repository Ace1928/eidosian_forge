from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.core import exceptions
def GetConditions(self):
    """Returns the resource conditions wrapped in condition.Conditions.

    Returns:
      A condition.Conditions object.
    """
    job_obj = self._resource_getter()
    if job_obj is None:
        return None
    conditions = job_obj.GetConditions(self._terminal_condition)
    self._PotentiallyUpdateInstanceCompletions(job_obj, conditions)
    return conditions