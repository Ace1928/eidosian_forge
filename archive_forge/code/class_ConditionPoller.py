from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.core import exceptions
class ConditionPoller(waiter.OperationPoller):
    """A poller for CloudRun resource creation or update.

  Takes in a reference to a StagedProgressTracker, and updates it with progress.
  """

    def __init__(self, resource_getter, tracker, dependencies=None, ready_message='Done.'):
        """Initialize the ConditionPoller.

    Start any unblocked stages in the tracker immediately.

    Arguments:
      resource_getter: function, returns a resource with conditions.
      tracker: a StagedProgressTracker to keep updated. It must contain a stage
        for each condition in the dependencies map, if the dependencies map is
        provided.  The stage represented by each key can only start when the set
        of conditions in the corresponding value have all completed. If a
        condition should be managed by this ConditionPoller but depends on
        nothing, it should map to an empty set. Conditions in the tracker but
        *not* managed by the ConditionPoller should not appear in the dict.
      dependencies: Dict[str, Set[str]], The dependencies between conditions
        that are managed by this ConditionPoller. The values are the set of
        conditions that must become true before the key begins being worked on
        by the server.  If the entire dependencies dict is None, the poller will
        assume that all keys in the tracker are relevant and none have
        dependencies.
      ready_message: str, message to display in header of tracker when
        conditions are ready.
    """
        self._dependencies = {k: set() for k in tracker}
        if dependencies is not None:
            for k in dependencies:
                self._dependencies[k] = {c for c in dependencies[k] if c in tracker and (not tracker.IsComplete(c))}
        self._resource_getter = resource_getter
        self._tracker = tracker
        self._resource_fail_type = exceptions.Error
        self._ready_message = ready_message
        self._StartUnblocked()

    def _IsBlocked(self, condition):
        return condition in self._dependencies and self._dependencies[condition]

    def IsDone(self, conditions):
        """Overrides.

    Args:
      conditions: A condition.Conditions object.

    Returns:
      a bool indicates whether `conditions` is terminal.
    """
        if conditions is None:
            return False
        return conditions.IsTerminal()

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

    def Poll(self, unused_ref):
        """Overrides.

    Args:
      unused_ref: A string representing the operation reference. Currently it
        must be 'deploy'.

    Returns:
      A condition.Conditions object.
    """
        conditions = self.GetConditions()
        if conditions is None or not conditions.IsFresh():
            return None
        conditions_message = conditions.DescriptiveMessage()
        self._tracker.UpdateHeaderMessage(conditions_message)
        self._PollTerminalSubconditions(conditions, conditions_message)
        terminal_condition = conditions.TerminalCondition()
        if conditions.IsReady():
            self._tracker.UpdateHeaderMessage(self._ready_message)
            if terminal_condition in self._dependencies:
                self._PossiblyCompleteStage(terminal_condition, None)
            self._tracker.Tick()
        elif conditions.IsFailed():
            if terminal_condition in self._dependencies:
                self._PossiblyFailStage(terminal_condition, None)
            raise self._resource_fail_type(conditions_message)
        return conditions

    def GetResource(self):
        return self._resource_getter()

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

    def _RecordConditionComplete(self, condition):
        """Take care of the internal-to-this-class bookkeeping stage complete."""
        for requirements in self._dependencies.values():
            requirements.discard(condition)

    def _PossiblyCompleteStage(self, condition, message):
        """Complete the stage if it's not already complete.

    Make sure the necessary internal bookkeeping is done.

    Args:
      condition: str, The name of the condition whose stage should be completed.
      message: str, The detailed message for the condition.

    Returns:
      bool: True if stage was completed, False if no action taken
    """
        if condition not in self._tracker or self._tracker.IsComplete(condition):
            return False
        if not self._tracker.IsRunning(condition):
            return False
        self._RecordConditionComplete(condition)
        self._StartUnblocked()
        self._tracker.CompleteStage(condition, message)
        return True

    def _StartUnblocked(self):
        """Call StartStage in the tracker for any not-started not-blocked tasks.

    Record the fact that they're started in our internal bookkeeping.
    """
        for c in self._dependencies:
            if c not in self._tracker:
                continue
            if self._tracker.IsWaiting(c) and (not self._IsBlocked(c)):
                self._tracker.StartStage(c)
        self._tracker.Tick()

    def _PossiblyFailStage(self, condition, message):
        """Possibly fail the stage.

    Args:
      condition: str, The name of the status whose stage failed.
      message: str, The detailed message for the condition.

    Raises:
      DeploymentFailedError: If the 'Ready' condition failed.
    """
        if condition not in self._tracker or self._tracker.IsComplete(condition):
            return
        self._tracker.FailStage(condition, self._resource_fail_type(message), message)

    def GetResult(self, conditions):
        """Overrides.

    Get terminal conditions as the polling result.

    Args:
      conditions: A condition.Conditions object.

    Returns:
      A condition.Conditions object.
    """
        return conditions

    def GetConditions(self):
        """Returns the resource conditions wrapped in condition.Conditions.

    Returns:
      A condition.Conditions object.
    """
        resource = self._resource_getter()
        if resource is None:
            return None
        return resource.conditions