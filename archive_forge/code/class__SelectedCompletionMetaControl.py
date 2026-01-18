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
class _SelectedCompletionMetaControl(UIControl):
    """
    Control that shows the meta information of the selected token.
    """

    def preferred_width(self, cli, max_available_width):
        """
        Report the width of the longest meta text as the preferred width of this control.

        It could be that we use less width, but this way, we're sure that the
        layout doesn't change when we select another completion (E.g. that
        completions are suddenly shown in more or fewer columns.)
        """
        if cli.current_buffer.complete_state:
            state = cli.current_buffer.complete_state
            return 2 + max((get_cwidth(c.display_meta) for c in state.current_completions))
        else:
            return 0

    def preferred_height(self, cli, width, max_available_height, wrap_lines):
        return 1

    def create_content(self, cli, width, height):
        tokens = self._get_tokens(cli)

        def get_line(i):
            return tokens
        return UIContent(get_line=get_line, line_count=1 if tokens else 0)

    def _get_tokens(self, cli):
        token = Token.Menu.Completions.MultiColumnMeta
        state = cli.current_buffer.complete_state
        if state and state.current_completion and state.current_completion.display_meta:
            return [(token, ' %s ' % state.current_completion.display_meta)]
        return []