import re
from typing import Dict, Any
class InlineState:
    """The state to save inline parser's tokens."""

    def __init__(self, env: Dict[str, Any]):
        self.env = env
        self.src = ''
        self.tokens = []
        self.in_image = False
        self.in_link = False
        self.in_emphasis = False
        self.in_strong = False

    def prepend_token(self, token: Dict[str, Any]):
        """Insert token before the last token."""
        self.tokens.insert(len(self.tokens) - 1, token)

    def append_token(self, token: Dict[str, Any]):
        """Add token to the end of token list."""
        self.tokens.append(token)

    def copy(self):
        """Create a copy of current state."""
        state = self.__class__(self.env)
        state.in_image = self.in_image
        state.in_link = self.in_link
        state.in_emphasis = self.in_emphasis
        state.in_strong = self.in_strong
        return state