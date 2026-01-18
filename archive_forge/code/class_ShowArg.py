from a buffer before the BufferControl will render it to the screen.
from __future__ import annotations
import re
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Callable, Hashable, cast
from prompt_toolkit.application.current import get_app
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.document import Document
from prompt_toolkit.filters import FilterOrBool, to_filter, vi_insert_multiple_mode
from prompt_toolkit.formatted_text import (
from prompt_toolkit.formatted_text.utils import fragment_list_len, fragment_list_to_text
from prompt_toolkit.search import SearchDirection
from prompt_toolkit.utils import to_int, to_str
from .utils import explode_text_fragments
class ShowArg(BeforeInput):
    """
    Display the 'arg' in front of the input.

    This was used by the `PromptSession`, but now it uses the
    `Window.get_line_prefix` function instead.
    """

    def __init__(self) -> None:
        super().__init__(self._get_text_fragments)

    def _get_text_fragments(self) -> StyleAndTextTuples:
        app = get_app()
        if app.key_processor.arg is None:
            return []
        else:
            arg = app.key_processor.arg
            return [('class:prompt.arg', '(arg: '), ('class:prompt.arg.text', str(arg)), ('class:prompt.arg', ') ')]

    def __repr__(self) -> str:
        return 'ShowArg()'