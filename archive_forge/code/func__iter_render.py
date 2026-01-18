from typing import Optional
from .core import BlockState
from .block_parser import BlockParser
from .inline_parser import InlineParser
def _iter_render(self, tokens, state):
    for tok in tokens:
        if 'children' in tok:
            children = self._iter_render(tok['children'], state)
            tok['children'] = list(children)
        elif 'text' in tok:
            text = tok.pop('text')
            tok['children'] = self.inline(text.strip(' \r\n\t\x0c'), state.env)
        yield tok