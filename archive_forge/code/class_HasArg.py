from __future__ import unicode_literals
from .base import Filter
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.key_binding.vi_state import InputMode as ViInputMode
from prompt_toolkit.cache import memoized
@memoized()
class HasArg(Filter):
    """
    Enable when the input processor has an 'arg'.
    """

    def __call__(self, cli):
        return cli.input_processor.arg is not None

    def __repr__(self):
        return 'HasArg()'