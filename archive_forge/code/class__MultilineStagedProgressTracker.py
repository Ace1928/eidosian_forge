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
class _MultilineStagedProgressTracker(_BaseStagedProgressTracker):
    """A context manager for telling the user about long-running progress.

  This class uses the core.console.multiline.ConsoleOutput interface for
  outputting. The header and each stage is defined as a message object
  contained by the ConsoleOutput message.
  """

    def __init__(self, *args, **kwargs):
        self._parser = parser.GetTypedTextParser()
        super(_MultilineStagedProgressTracker, self).__init__(*args, **kwargs)

    def UpdateHeaderMessage(self, message):
        self._header_stage.message = message

    def _UpdateHeaderMessage(self, prefix):
        message = prefix + self._message
        if self._header_stage.message:
            message += ' ' + self._header_stage.message
        self._UpdateMessage(self._header_message, message)

    def _UpdateStageTickMark(self, stage, tick_mark=''):
        prefix = self._GenerateStagePrefix(stage.status, tick_mark)
        message = stage.header
        if stage.message:
            message += ' ' + stage.message
        self._UpdateMessage(self._stage_messages[stage], prefix + message)

    def _UpdateMessage(self, stage, message):
        message = self._parser.ParseTypedTextToString(message)
        self._console_output.UpdateMessage(stage, message)

    def _AddMessage(self, message, indentation_level=0):
        message = self._parser.ParseTypedTextToString(message)
        return self._console_output.AddMessage(message, indentation_level=indentation_level)

    def _NotifyUninterruptableError(self):
        with self._lock:
            self.UpdateHeaderMessage('This operation cannot be cancelled.')
        self.Tick()

    def _SetupExitOutput(self):
        """Sets up output to print out the closing line."""
        return self._console_output.AddMessage('')

    def _PrintExitOutput(self, aborted=False, warned=False, failed=False):
        """Handles the final output for the progress tracker."""
        output_message = self._SetupExitOutput()
        if aborted:
            msg = self._aborted_message or 'Aborted.'
            self._header_stage.status = StageCompletionStatus.FAILED
        elif failed:
            msg = self._failure_message or 'Failed.'
            self._header_stage.status = StageCompletionStatus.FAILED
        elif warned:
            msg = self._warning_message or 'Completed with warnings:'
            self._header_stage.status = StageCompletionStatus.FAILED
        else:
            msg = self._success_message or 'Done.'
            self._header_stage.status = StageCompletionStatus.SUCCESS
        if self._done_message_callback:
            msg += ' ' + self._done_message_callback()
        self._UpdateMessage(output_message, msg)
        self._Print(self._symbols.interrupted)

    def _SetupOutput(self):
        self._maintain_queue = False
        self._console_output = multiline.MultilineConsoleOutput(self._stream)
        self._header_message = self._AddMessage(self._message)
        self._header_stage = Stage('')
        self._header_stage.status = StageCompletionStatus.RUNNING
        self._stage_messages = dict()
        for stage in self._stages.values():
            self._stage_messages[stage] = self._AddMessage(stage.header, indentation_level=1)
            self._UpdateStageTickMark(stage)
        self._console_output.UpdateConsole()

    def _GenerateStagePrefix(self, stage_status, tick_mark):
        if stage_status == StageCompletionStatus.NOT_STARTED:
            tick_mark = self._symbols.not_started
        elif stage_status == StageCompletionStatus.SUCCESS:
            tick_mark = self._symbols.success
        elif stage_status == StageCompletionStatus.FAILED:
            tick_mark = self._symbols.failed
        elif stage_status == StageCompletionStatus.INTERRUPTED:
            tick_mark = self._symbols.interrupted
        return tick_mark + ' ' * (self._symbols.prefix_length - len(tick_mark))

    def _FailStage(self, stage, exception, message=None):
        """Informs the progress tracker that this stage has failed."""
        self._UpdateStageTickMark(stage)
        if exception:
            for other_stage in self._stages.values():
                if other_stage != stage and other_stage.status == StageCompletionStatus.RUNNING:
                    other_stage.status = StageCompletionStatus.INTERRUPTED
                other_stage._is_done = True

    def _CompleteStage(self, stage):
        self._UpdateStageTickMark(stage)

    def _CompleteStageWithWarnings(self, stage, warning_messages):
        self._UpdateStageTickMark(stage)

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
                self._Print(self._GetTickMark(self._ticks))
        return self._done

    def _Print(self, tick_mark=''):
        """Prints an update containing message to the output stream.

    Args:
      tick_mark: str, suffix of message
    """
        if not self._output_enabled:
            return
        header_prefix = self._GenerateStagePrefix(self._header_stage.status, tick_mark)
        self._UpdateHeaderMessage(header_prefix)
        for key in self._running_stages:
            self._UpdateStageTickMark(self[key], tick_mark)
        self._console_output.UpdateConsole()