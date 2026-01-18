import collections
import re
from colorsys import hls_to_rgb
from .parser import parse_one_component_value
Parse a list of tokens (typically the content of a function token)
    as arguments made of a single token each, separated by mandatory commas,
    with optional white space around each argument.

    return the argument list without commas or white space;
    or None if the function token content do not match the description above.

    