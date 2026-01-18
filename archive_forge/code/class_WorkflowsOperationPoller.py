from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.api_lib.workflows import codes
class WorkflowsOperationPoller(waiter.OperationPoller):
    """Implementation of OperationPoller for Workflows Operations."""

    def __init__(self, workflows, operations, workflow_ref):
        """Creates the poller.

    Args:
      workflows: the Workflows API client used to get the resource after
        operation is complete.
      operations: the Operations API client used to poll for the operation.
      workflow_ref: a reference to a workflow that is the subject of this
        operation.
    """
        self.workflows = workflows
        self.operations = operations
        self.workflow_ref = workflow_ref

    def IsDone(self, operation):
        """Overrides."""
        if operation.done:
            if operation.error:
                raise waiter.OperationError(_ExtractErrorMessage(operation.error))
            return True
        return False

    def Poll(self, operation_ref):
        """Overrides."""
        return self.operations.Get(operation_ref)

    def GetResult(self, operation):
        """Overrides."""
        return self.workflows.Get(self.workflow_ref)