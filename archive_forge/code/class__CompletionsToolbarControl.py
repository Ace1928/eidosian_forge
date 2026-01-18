from __future__ import annotations
from typing import Any
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.enums import SYSTEM_BUFFER
from prompt_toolkit.filters import (
from prompt_toolkit.formatted_text import (
from prompt_toolkit.key_binding.key_bindings import (
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.key_binding.vi_state import InputMode
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import ConditionalContainer, Container, Window
from prompt_toolkit.layout.controls import (
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.processors import BeforeInput
from prompt_toolkit.lexers import SimpleLexer
from prompt_toolkit.search import SearchDirection
class _CompletionsToolbarControl(UIControl):

    def create_content(self, width: int, height: int) -> UIContent:
        all_fragments: StyleAndTextTuples = []
        complete_state = get_app().current_buffer.complete_state
        if complete_state:
            completions = complete_state.completions
            index = complete_state.complete_index
            content_width = width - 6
            cut_left = False
            cut_right = False
            fragments: StyleAndTextTuples = []
            for i, c in enumerate(completions):
                if fragment_list_len(fragments) + len(c.display_text) >= content_width:
                    if i <= (index or 0):
                        fragments = []
                        cut_left = True
                    else:
                        cut_right = True
                        break
                fragments.extend(to_formatted_text(c.display_text, style='class:completion-toolbar.completion.current' if i == index else 'class:completion-toolbar.completion'))
                fragments.append(('', ' '))
            fragments.append(('', ' ' * (content_width - fragment_list_len(fragments))))
            fragments = fragments[:content_width]
            all_fragments.append(('', ' '))
            all_fragments.append(('class:completion-toolbar.arrow', '<' if cut_left else ' '))
            all_fragments.append(('', ' '))
            all_fragments.extend(fragments)
            all_fragments.append(('', ' '))
            all_fragments.append(('class:completion-toolbar.arrow', '>' if cut_right else ' '))
            all_fragments.append(('', ' '))

        def get_line(i: int) -> StyleAndTextTuples:
            return all_fragments
        return UIContent(get_line=get_line, line_count=1)