from __future__ import unicode_literals
from ..enums import IncrementalSearchDirection
from .processors import BeforeInput
from .lexers import SimpleLexer
from .dimension import LayoutDimension
from .controls import BufferControl, TokenListControl, UIControl, UIContent
from .containers import Window, ConditionalContainer
from .screen import Char
from .utils import token_list_len
from prompt_toolkit.enums import SEARCH_BUFFER, SYSTEM_BUFFER
from prompt_toolkit.filters import HasFocus, HasArg, HasCompletions, HasValidationError, HasSearch, Always, IsDone
from prompt_toolkit.token import Token
class CompletionsToolbarControl(UIControl):
    token = Token.Toolbar.Completions

    def create_content(self, cli, width, height):
        complete_state = cli.current_buffer.complete_state
        if complete_state:
            completions = complete_state.current_completions
            index = complete_state.complete_index
            content_width = width - 6
            cut_left = False
            cut_right = False
            tokens = []
            for i, c in enumerate(completions):
                if token_list_len(tokens) + len(c.display) >= content_width:
                    if i <= (index or 0):
                        tokens = []
                        cut_left = True
                    else:
                        cut_right = True
                        break
                tokens.append((self.token.Completion.Current if i == index else self.token.Completion, c.display))
                tokens.append((self.token, ' '))
            tokens.append((self.token, ' ' * (content_width - token_list_len(tokens))))
            tokens = tokens[:content_width]
            all_tokens = [(self.token, ' '), (self.token.Arrow, '<' if cut_left else ' '), (self.token, ' ')] + tokens + [(self.token, ' '), (self.token.Arrow, '>' if cut_right else ' '), (self.token, ' ')]
        else:
            all_tokens = []

        def get_line(i):
            return all_tokens
        return UIContent(get_line=get_line, line_count=1)