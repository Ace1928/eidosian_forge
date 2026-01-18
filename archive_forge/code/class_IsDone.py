from __future__ import unicode_literals
from .base import Filter
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.key_binding.vi_state import InputMode as ViInputMode
from prompt_toolkit.cache import memoized
@memoized()
class IsDone(Filter):
    """
    True when the CLI is returning, aborting or exiting.
    """

    def __call__(self, cli):
        return cli.is_done

    def __repr__(self):
        return 'IsDone()'