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
class _MergedProcessor(Processor):
    """
    Processor that groups multiple other `Processor` objects, but exposes an
    API as if it is one `Processor`.
    """

    def __init__(self, processors: list[Processor]):
        self.processors = processors

    def apply_transformation(self, ti: TransformationInput) -> Transformation:
        source_to_display_functions = [ti.source_to_display]
        display_to_source_functions = []
        fragments = ti.fragments

        def source_to_display(i: int) -> int:
            """Translate x position from the buffer to the x position in the
            processor fragments list."""
            for f in source_to_display_functions:
                i = f(i)
            return i
        for p in self.processors:
            transformation = p.apply_transformation(TransformationInput(ti.buffer_control, ti.document, ti.lineno, source_to_display, fragments, ti.width, ti.height))
            fragments = transformation.fragments
            display_to_source_functions.append(transformation.display_to_source)
            source_to_display_functions.append(transformation.source_to_display)

        def display_to_source(i: int) -> int:
            for f in reversed(display_to_source_functions):
                i = f(i)
            return i
        del source_to_display_functions[:1]
        return Transformation(fragments, source_to_display, display_to_source)