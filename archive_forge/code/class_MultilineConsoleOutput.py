from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import threading
from googlecloudsdk.core.console import console_attr
import six
class MultilineConsoleOutput(ConsoleOutput):
    """An implementation of ConsoleOutput which supports multiline updates.

  This means all messages can be updated and actually have their output
  be updated on the terminal. The main difference between this class and
  the simple suffix version is that updates here are updates to the entire
  message as this provides more flexibility.

  This class accepts messages containing ANSI escape codes. The width
  calculations will be handled correctly currently only in this class.
  """

    def __init__(self, stream):
        """Constructor.

    Args:
      stream: The output stream to write to.
    """
        self._stream = stream
        self._messages = []
        self._last_print_index = 0
        self._lock = threading.Lock()
        self._last_total_lines = 0
        self._may_have_update = False
        super(MultilineConsoleOutput, self).__init__()

    def AddMessage(self, message, indentation_level=0):
        """Adds a MultilineConsoleMessage to the MultilineConsoleOutput object.

    Args:
      message: str, The message that will be displayed.
      indentation_level: int, The indentation level of the message. Each
        indentation is represented by two spaces.

    Returns:
      MultilineConsoleMessage, a message object that can be used to dynamically
      change the printed message.
    """
        with self._lock:
            return self._AddMessage(message, indentation_level=indentation_level)

    def _AddMessage(self, message, indentation_level=0):
        self._may_have_update = True
        console_message = MultilineConsoleMessage(message, self._stream, indentation_level=indentation_level)
        self._messages.append(console_message)
        return console_message

    def UpdateMessage(self, message, new_message):
        """Updates the message of the given MultilineConsoleMessage."""
        if not message:
            raise ValueError('A message must be passed.')
        if message not in self._messages:
            raise ValueError('The given message does not belong to this output object.')
        with self._lock:
            message._UpdateMessage(new_message)
            self._may_have_update = True

    def UpdateConsole(self):
        with self._lock:
            self._UpdateConsole()

    def _GetAnsiCursorUpSequence(self, num_lines):
        """Returns an ANSI control sequences that moves the cursor up num_lines."""
        return '\x1b[{}A'.format(num_lines)

    def _UpdateConsole(self):
        """Updates the console output to show any updated or added messages."""
        if not self._may_have_update:
            return
        if self._last_total_lines:
            self._stream.write(self._GetAnsiCursorUpSequence(self._last_total_lines))
        total_lines = 0
        force_print_rest = False
        for message in self._messages:
            num_lines = message.num_lines
            total_lines += num_lines
            if message.has_update or force_print_rest:
                force_print_rest |= message.num_lines_changed
                message.Print()
            else:
                self._stream.write('\n' * num_lines)
        self._last_total_lines = total_lines
        self._may_have_update = False