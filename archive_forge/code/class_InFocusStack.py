from __future__ import unicode_literals
from .base import Filter
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.key_binding.vi_state import InputMode as ViInputMode
from prompt_toolkit.cache import memoized
@memoized()
class InFocusStack(Filter):
    """
    Enable when this buffer appears on the focus stack.
    """

    def __init__(self, buffer_name):
        self._buffer_name = buffer_name

    @property
    def buffer_name(self):
        """ The given buffer name. (Read-only) """
        return self._buffer_name

    def __call__(self, cli):
        return self.buffer_name in cli.buffers.focus_stack

    def __repr__(self):
        return 'InFocusStack(%r)' % self.buffer_name