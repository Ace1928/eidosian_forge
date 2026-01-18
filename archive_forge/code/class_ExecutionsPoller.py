from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.api_lib.workflows import codes
class ExecutionsPoller(waiter.OperationPoller):
    """Implementation of OperationPoller for Workflows Executions."""

    def __init__(self, workflow_execution):
        """Creates the execution poller.

    Args:
      workflow_execution: the Workflows Executions API client used to get the
        execution resource.
    """
        self.workflow_execution = workflow_execution

    def IsDone(self, execution):
        """Overrides."""
        return execution.state.name != 'ACTIVE' and execution.state.name != 'QUEUED'

    def Poll(self, execution_ref):
        """Overrides."""
        return self.workflow_execution.Get(execution_ref)

    def GetResult(self, execution):
        """Overrides."""
        return execution