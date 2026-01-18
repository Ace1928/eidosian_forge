from __future__ import unicode_literals
from prompt_toolkit.filters import to_cli_filter
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Point, Screen, WritePosition
from prompt_toolkit.output import Output
from prompt_toolkit.styles import Style
from prompt_toolkit.token import Token
from prompt_toolkit.utils import is_windows
from six.moves import range
class _TokenToAttrsCache(dict):
    """
    A cache structure that maps Pygments Tokens to :class:`.Attr`.
    (This is an important speed up.)
    """

    def __init__(self, get_style_for_token):
        self.get_style_for_token = get_style_for_token

    def __missing__(self, token):
        try:
            result = self.get_style_for_token(token)
        except KeyError:
            result = None
        self[token] = result
        return result