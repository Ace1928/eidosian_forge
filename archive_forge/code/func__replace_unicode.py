import sys
import re
import operator
import typing
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union
def _replace_unicode(match: 're.Match[str]') -> str:
    codepoint = int(match.group(1), 16)
    if codepoint > sys.maxunicode:
        codepoint = 65533
    return chr(codepoint)