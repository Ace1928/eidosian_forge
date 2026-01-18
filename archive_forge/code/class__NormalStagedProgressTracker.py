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
class _NormalStagedProgressTracker(_BaseStagedProgressTracker):
    """A context manager for telling the user about long-running progress.

  This class uses the core.console.multiline.ConsoleOutput interface for
  outputting. The header and each stage is defined as a message object
  contained by the ConsoleOutput message.
  """

    def __init__(self, *args, **kwargs):
        self._running_stages_queue = []
        self._stage_being_displayed = None
        super(_NormalStagedProgressTracker, self).__init__(*args, **kwargs)

    def _SetupOutput(self):
        self._console_output = multiline.SimpleSuffixConsoleOutput(self._stream)
        self._header_message = self._console_output.AddMessage(self._message)
        self._current_stage_message = self._header_message

    def _FailStage(self, stage, failure_exception, message=None):
        for running_stage in self._running_stages_queue:
            if stage != running_stage:
                running_stage.status = StageCompletionStatus.INTERRUPTED
            running_stage._is_done = True

    def _StartStage(self, stage):
        """Informs the progress tracker that this stage has started."""
        self._running_stages_queue.append(stage)
        if self._stage_being_displayed is None:
            self._LoadNextStageForDisplay()

    def _LoadNextStageForDisplay(self):
        if self._running_stages_queue:
            self._stage_being_displayed = self._running_stages_queue[0]
            self._SetUpOutputForStage(self._stage_being_displayed)
            return True

    def Tick(self):
        """Give a visual indication to the user that some progress has been made.

    Output is sent to sys.stderr. Nothing is shown if output is not a TTY.
    This method also handles loading new stages and flushing out completed
    stages.

    Returns:
      Whether progress has completed.
    """
        with self._lock:
            if not self._done:
                self._ticks += 1
                if self._stage_being_displayed is None:
                    self._LoadNextStageForDisplay()
                else:
                    while self._running_stages_queue and self._running_stages_queue[0].is_done:
                        completed_stage = self._running_stages_queue.pop(0)
                        self._completed_stages.append(completed_stage.key)
                        completion_status = self._GetStagedCompletedSuffix(self._stage_being_displayed.status)
                        self._Print(completion_status)
                        if not self._LoadNextStageForDisplay():
                            self._stage_being_displayed = None
                if self._stage_being_displayed:
                    self._Print(self._GetTickMark(self._ticks))
        return self._done

    def _PrintExitOutput(self, aborted=False, warned=False, failed=False):
        """Handles the final output for the progress tracker."""
        self._SetupExitOutput()
        if aborted:
            msg = self._aborted_message or 'Aborted.'
        elif failed:
            msg = self._failure_message or 'Failed.'
        elif warned:
            msg = self._warning_message or 'Completed with warnings:'
        else:
            msg = self._success_message or 'Done.'
        if self._done_message_callback:
            msg += ' ' + self._done_message_callback()
        self._Print(msg + '\n')

    def _SetupExitOutput(self):
        """Sets up output to print out the closing line."""
        self._current_stage_message = self._console_output.AddMessage('')

    def _HandleUncaughtException(self, exc_value):
        if isinstance(exc_value, console_io.OperationCancelledError):
            self._Print('aborted by ctrl-c')
            self._PrintExitOutput(aborted=True)
        else:
            self._Print(self._GetStagedCompletedSuffix(StageCompletionStatus.FAILED))
            self._PrintExitOutput(failed=True)

    def _SetUpOutputForStage(self, stage):

        def _FormattedCallback():
            if stage.message:
                return ' ' + stage.message + '...'
            return None
        self._current_stage_message = self._console_output.AddMessage(stage.header, indentation_level=1, detail_message_callback=_FormattedCallback)

    def _Print(self, message=''):
        """Prints an update containing message to the output stream.

    Args:
      message: str, suffix of message
    """
        if not self._output_enabled:
            return
        if self._current_stage_message:
            self._console_output.UpdateMessage(self._current_stage_message, message)
            self._console_output.UpdateConsole()