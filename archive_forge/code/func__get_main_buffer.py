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
def _get_main_buffer(self, buffer_control: BufferControl) -> BufferControl | None:
    from prompt_toolkit.layout.controls import BufferControl
    prev_control = get_app().layout.search_target_buffer_control
    if isinstance(prev_control, BufferControl) and prev_control.search_buffer_control == buffer_control:
        return prev_control
    return None