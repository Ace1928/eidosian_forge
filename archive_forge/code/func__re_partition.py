from __future__ import annotations
import argparse
import io
import keyword
import re
import sys
import tokenize
from typing import Generator
from typing import Iterable
from typing import NamedTuple
from typing import Pattern
from typing import Sequence
def _re_partition(regex: Pattern[str], s: str) -> tuple[str, str, str]:
    match = regex.search(s)
    if match:
        return (s[:match.start()], s[slice(*match.span())], s[match.end():])
    else:
        return (s, '', '')