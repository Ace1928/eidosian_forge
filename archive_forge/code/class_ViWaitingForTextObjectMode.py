from __future__ import unicode_literals
from .base import Filter
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.key_binding.vi_state import InputMode as ViInputMode
from prompt_toolkit.cache import memoized
@memoized()
class ViWaitingForTextObjectMode(Filter):

    def __call__(self, cli):
        if cli.editing_mode != EditingMode.VI:
            return False
        return cli.vi_state.operator_func is not None

    def __repr__(self):
        return 'ViWaitingForTextObjectMode()'