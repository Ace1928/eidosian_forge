from collections import Counter
from textwrap import dedent
from kombu.utils.encoding import bytes_to_str, safe_str
class DOT:
    """Constants related to the dot format."""
    HEAD = dedent('\n        {IN}{type} {id} {{\n        {INp}graph [{attrs}]\n    ')
    ATTR = '{name}={value}'
    NODE = '{INp}"{0}" [{attrs}]'
    EDGE = '{INp}"{0}" {dir} "{1}" [{attrs}]'
    ATTRSEP = ', '
    DIRS = {'graph': '--', 'digraph': '->'}
    TAIL = '{IN}}}'