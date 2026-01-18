import collections
import re
from colorsys import hls_to_rgb
from .parser import parse_one_component_value
def _parse_comma_separated(tokens):
    """Parse a list of tokens (typically the content of a function token)
    as arguments made of a single token each, separated by mandatory commas,
    with optional white space around each argument.

    return the argument list without commas or white space;
    or None if the function token content do not match the description above.

    """
    tokens = [token for token in tokens if token.type not in ('whitespace', 'comment')]
    if not tokens:
        return []
    if len(tokens) % 2 == 1 and all((token == ',' for token in tokens[1::2])):
        return tokens[::2]