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
class TransformationInput:
    """
    :param buffer_control: :class:`.BufferControl` instance.
    :param lineno: The number of the line to which we apply the processor.
    :param source_to_display: A function that returns the position in the
        `fragments` for any position in the source string. (This takes
        previous processors into account.)
    :param fragments: List of fragments that we can transform. (Received from the
        previous processor.)
    """

    def __init__(self, buffer_control: BufferControl, document: Document, lineno: int, source_to_display: SourceToDisplay, fragments: StyleAndTextTuples, width: int, height: int) -> None:
        self.buffer_control = buffer_control
        self.document = document
        self.lineno = lineno
        self.source_to_display = source_to_display
        self.fragments = fragments
        self.width = width
        self.height = height

    def unpack(self) -> tuple[BufferControl, Document, int, SourceToDisplay, StyleAndTextTuples, int, int]:
        return (self.buffer_control, self.document, self.lineno, self.source_to_display, self.fragments, self.width, self.height)