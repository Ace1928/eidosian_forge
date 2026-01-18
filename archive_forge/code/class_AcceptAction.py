from __future__ import unicode_literals
from .auto_suggest import AutoSuggest
from .clipboard import ClipboardData
from .completion import Completer, Completion, CompleteEvent
from .document import Document
from .enums import IncrementalSearchDirection
from .filters import to_simple_filter
from .history import History, InMemoryHistory
from .search_state import SearchState
from .selection import SelectionType, SelectionState, PasteMode
from .utils import Event
from .cache import FastDictCache
from .validation import ValidationError
from six.moves import range
import os
import re
import six
import subprocess
import tempfile
class AcceptAction(object):
    """
    What to do when the input is accepted by the user.
    (When Enter was pressed in the command line.)

    :param handler: (optional) A callable which takes a
        :class:`~prompt_toolkit.interface.CommandLineInterface` and
        :class:`~prompt_toolkit.document.Document`. It is called when the user
        accepts input.
    """

    def __init__(self, handler=None):
        assert handler is None or callable(handler)
        self.handler = handler

    @classmethod
    def run_in_terminal(cls, handler, render_cli_done=False):
        """
        Create an :class:`.AcceptAction` that runs the given handler in the
        terminal.

        :param render_cli_done: When True, render the interface in the 'Done'
                state first, then execute the function. If False, erase the
                interface instead.
        """

        def _handler(cli, buffer):
            cli.run_in_terminal(lambda: handler(cli, buffer), render_cli_done=render_cli_done)
        return AcceptAction(handler=_handler)

    @property
    def is_returnable(self):
        """
        True when there is something handling accept.
        """
        return bool(self.handler)

    def validate_and_handle(self, cli, buffer):
        """
        Validate buffer and handle the accept action.
        """
        if buffer.validate():
            if self.handler:
                self.handler(cli, buffer)
            buffer.append_to_history()