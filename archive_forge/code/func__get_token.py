import re
from typing import List, Optional, Tuple
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