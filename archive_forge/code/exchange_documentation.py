from __future__ import annotations
import re
from kombu.utils.text import escape_regex
Match regular expression (cached).

        Same as :func:`re.match`, except the regex is compiled and cached,
        then reused on subsequent matches with the same pattern.
        