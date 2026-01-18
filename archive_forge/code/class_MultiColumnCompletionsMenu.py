from __future__ import unicode_literals
from six.moves import zip_longest, range
from prompt_toolkit.filters import HasCompletions, IsDone, Condition, to_cli_filter
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.token import Token
from prompt_toolkit.utils import get_cwidth
from .containers import Window, HSplit, ConditionalContainer, ScrollOffsets
from .controls import UIControl, UIContent
from .dimension import LayoutDimension
from .margins import ScrollbarMargin
from .screen import Point, Char
import math
class MultiColumnCompletionsMenu(HSplit):
    """
    Container that displays the completions in several columns.
    When `show_meta` (a :class:`~prompt_toolkit.filters.CLIFilter`) evaluates
    to True, it shows the meta information at the bottom.
    """

    def __init__(self, min_rows=3, suggested_max_column_width=30, show_meta=True, extra_filter=True):
        show_meta = to_cli_filter(show_meta)
        extra_filter = to_cli_filter(extra_filter)
        full_filter = HasCompletions() & ~IsDone() & extra_filter
        any_completion_has_meta = Condition(lambda cli: any((c.display_meta for c in cli.current_buffer.complete_state.current_completions)))
        completions_window = ConditionalContainer(content=Window(content=MultiColumnCompletionMenuControl(min_rows=min_rows, suggested_max_column_width=suggested_max_column_width), width=LayoutDimension(min=8), height=LayoutDimension(min=1)), filter=full_filter)
        meta_window = ConditionalContainer(content=Window(content=_SelectedCompletionMetaControl()), filter=show_meta & full_filter & any_completion_has_meta)
        super(MultiColumnCompletionsMenu, self).__init__([completions_window, meta_window])