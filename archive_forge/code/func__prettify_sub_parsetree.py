from __future__ import annotations
from collections import OrderedDict
from types import MethodType
from typing import (
from pyparsing import ParserElement, ParseResults, TokenConverter, originalTextFor
from rdflib.term import BNode, Identifier, Variable
from rdflib.plugins.sparql.sparql import NotBoundError, SPARQLError  # noqa: E402
def _prettify_sub_parsetree(t: Union[Identifier, CompValue, set, list, dict, Tuple, bool, None], indent: str='', depth: int=0) -> str:
    out: List[str] = []
    if isinstance(t, CompValue):
        out.append('%s%s> %s:\n' % (indent, '  ' * depth, t.name))
        for k, v in t.items():
            out.append('%s%s- %s:\n' % (indent, '  ' * (depth + 1), k))
            out.append(_prettify_sub_parsetree(v, indent, depth + 2))
    elif isinstance(t, dict):
        for k, v in t.items():
            out.append('%s%s- %s:\n' % (indent, '  ' * (depth + 1), k))
            out.append(_prettify_sub_parsetree(v, indent, depth + 2))
    elif isinstance(t, list):
        for e in t:
            out.append(_prettify_sub_parsetree(e, indent, depth + 1))
    else:
        out.append('%s%s- %r\n' % (indent, '  ' * depth, t))
    return ''.join(out)