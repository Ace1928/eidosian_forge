import re
from typing import Optional, List, Dict, Any, Match
from .core import Parser, InlineState
from .util import (
from .helpers import (
def __parse_link_token(self, is_image, text, attrs, state):
    new_state = state.copy()
    new_state.src = text
    if is_image:
        new_state.in_image = True
        token = {'type': 'image', 'children': self.render(new_state), 'attrs': attrs}
    else:
        new_state.in_link = True
        token = {'type': 'link', 'children': self.render(new_state), 'attrs': attrs}
    return token