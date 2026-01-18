from typing import (
import re
from .termformatconstants import (
Returns processed text, the next token, and unprocessed text

    >>> front, d, rest = peel_off_esc_code('some[2Astuff')
    >>> front, rest
    ('some', 'stuff')
    >>> d == {'numbers': [2], 'command': 'A', 'intermed': '', 'private': '', 'csi': '\x1b[', 'seq': '\x1b[2A'}
    True
    