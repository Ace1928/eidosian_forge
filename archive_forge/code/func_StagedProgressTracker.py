from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import os
import signal
import sys
import threading
import time
import enum
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import multiline
from googlecloudsdk.core.console.style import parser
import six
def StagedProgressTracker(message, stages, tracker_id=None, autotick=True, tick_delay=0.1, interruptable=True, done_message_callback=None, success_message=None, warning_message=None, failure_message=None, aborted_message=console_io.OperationCancelledError.DEFAULT_MESSAGE, suppress_output=False):
    """A progress tracker for performing actions with multiple stages.

  The progress tracker is a context manager. To start displaying information
  about a running stage, call StartStage within the staged progress tracker
  context. To update the message of a stage, use UpdateStage. When a stage is
  completed/failed there are CompleteStage and FailStage methods on the
  tracker as well.

  Note that stages do not need to be started/completed in order. In
  non-multiline (the only supported mode) output mode, the displayed stage will
  be the earliest started stage that has not been completed.

  Example Usage:
    stages = [
      Stage('Getting bread...', key='bread'),
      Stage('Getting peanut butter...', key='pb'),
      Stage('Making sandwich...', key='make')]
    with StagedProgressTracker(
        'Making sandwich...',
        stages,
        success_message='Time to eat!',
        failure_message='Time to order delivery..!',
        tracker_id='meta.make_sandwich') as tracker:
      tracker.StartStage('bread')
      # Go to pantry
      tracker.UpdateStage('bread', 'Looking for bread in the pantry')
      # Get bread
      tracker.CompleteStage('bread', 'Got some whole wheat bread!')

      tracker.StartStage('pb')
      # Look for peanut butter
      if pb_not_found:
        error = exceptions.NoPeanutButterError('So sad!')
        tracker.FailStage('pb', error)
      elif pb_not_organic:
        tracker.CompleteStageWithWarning('pb', 'The pb is not organic!')
      else:
        tracker.CompleteStage('bread', 'Got some organic pb!')

  Args:
    message: str, The message to show next to the spinner.
    stages: list[Stage], A list of stages for the progress tracker to run. Once
      you pass the stages to a StagedProgressTracker, they're owned by the
      tracker and you should not mutate them.
    tracker_id: str The ID of this tracker that will be used for metrics.
    autotick: bool, True to have the spinner tick on its own. Otherwise, you
      need to call Tick() explicitly to move the spinner.
    tick_delay: float, The amount of time to wait between ticks, in second.
    interruptable: boolean, True if the user can ctrl-c the operation. If so,
      it will stop and will report as aborted. If False,
    done_message_callback: func, A callback to get a more detailed done message.
    success_message: str, A message to display on success of all tasks.
    warning_message: str, A message to display when no task fails but one or
      more tasks complete with a warning and none fail.
    failure_message: str, A message to display on failure of a task.
    aborted_message: str, A custom message to put in the exception when it is
      cancelled by the user.
    suppress_output: bool, True to suppress output from the tracker.

  Returns:
    The progress tracker.
  """
    style = properties.VALUES.core.interactive_ux_style.Get()
    if suppress_output or style == properties.VALUES.core.InteractiveUXStyles.OFF.name:
        return NoOpStagedProgressTracker(stages, interruptable, aborted_message)
    elif style == properties.VALUES.core.InteractiveUXStyles.TESTING.name:
        return _StubStagedProgressTracker(message, stages, interruptable, aborted_message)
    else:
        is_tty = console_io.IsInteractive(error=True)
        if is_tty:
            if console_attr.ConsoleAttr().SupportsAnsi():
                tracker_cls = _MultilineStagedProgressTracker
            else:
                tracker_cls = _NormalStagedProgressTracker
        else:
            tracker_cls = _NonInteractiveStagedProgressTracker
        return tracker_cls(message, stages, success_message, warning_message, failure_message, autotick, tick_delay, interruptable, aborted_message, tracker_id, done_message_callback)