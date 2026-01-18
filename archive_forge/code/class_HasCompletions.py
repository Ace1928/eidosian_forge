from __future__ import unicode_literals
from .base import Filter
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.key_binding.vi_state import InputMode as ViInputMode
from prompt_toolkit.cache import memoized
@memoized()
class HasCompletions(Filter):
    """
    Enable when the current buffer has completions.
    """

    def __call__(self, cli):
        return cli.current_buffer.complete_state is not None

    def __repr__(self):
        return 'HasCompletions()'