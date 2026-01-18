from __future__ import absolute_import
import sys
import re
import itertools as _itertools
from codecs import BOM_UTF8
from typing import NamedTuple, Tuple, Iterator, Iterable, List, Dict, \
from parso.python.token import PythonTokenTypes
from parso.utils import split_lines, PythonVersionInfo, parse_version_string
class TokenCollection(NamedTuple):
    pseudo_token: Pattern
    single_quoted: Set[str]
    triple_quoted: Set[str]
    endpats: Dict[str, Pattern]
    whitespace: Pattern
    fstring_pattern_map: Dict[str, str]
    always_break_tokens: Tuple[str]