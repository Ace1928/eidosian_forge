import re
from typing import List, Optional, Tuple
class Splitter:

    def __init__(self, command_line, single_quotes_allowed):
        self.seq = _PushbackSequence(command_line)
        self.allowed_quote_chars = '"'
        if single_quotes_allowed:
            self.allowed_quote_chars += "'"

    def __iter__(self):
        return self

    def __next__(self):
        quoted, token = self._get_token()
        if token is None:
            raise StopIteration
        return (quoted, token)
    next = __next__

    def _get_token(self) -> Tuple[bool, Optional[str]]:
        self.quoted = False
        self.token: List[str] = []
        state = _Whitespace()
        for next_char in self.seq:
            state = state.process(next_char, self)
            if state is None:
                break
        if state is not None and hasattr(state, 'finish'):
            state.finish(self)
        result: Optional[str] = ''.join(self.token)
        if not self.quoted and result == '':
            result = None
        return (self.quoted, result)