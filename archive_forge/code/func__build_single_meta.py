import inspect
import re
import typing as T
from collections import OrderedDict, namedtuple
from enum import IntEnum
from .common import (
@staticmethod
def _build_single_meta(section: Section, desc: str) -> DocstringMeta:
    if section.key in RETURNS_KEYWORDS | YIELDS_KEYWORDS:
        return DocstringReturns(args=[section.key], description=desc, type_name=None, is_generator=section.key in YIELDS_KEYWORDS)
    if section.key in RAISES_KEYWORDS:
        return DocstringRaises(args=[section.key], description=desc, type_name=None)
    if section.key in EXAMPLES_KEYWORDS:
        return DocstringExample(args=[section.key], snippet=None, description=desc)
    if section.key in PARAM_KEYWORDS:
        raise ParseError('Expected paramenter name.')
    return DocstringMeta(args=[section.key], description=desc)