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
def _show_meta(self, complete_state):
    """
        Return ``True`` if we need to show a column with meta information.
        """
    return any((c.display_meta for c in complete_state.current_completions))