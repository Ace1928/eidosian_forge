from __future__ import annotations
from typing import (
from rdflib.store import Store
from rdflib.util import _coalesce
def __add_triple_context(self, triple: '_TripleType', triple_exists: bool, context: Optional['_ContextType'], quoted: bool) -> None:
    """add the given context to the set of contexts for the triple"""
    ctx = self.__ctx_to_str(context)
    quoted = bool(quoted)
    if triple_exists:
        try:
            triple_context = self.__tripleContexts[triple]
        except KeyError:
            triple_context = self.__tripleContexts[triple] = self.__defaultContexts.copy()
        triple_context[ctx] = quoted
        if not quoted:
            triple_context[None] = quoted
    elif quoted:
        triple_context = self.__tripleContexts[triple] = {ctx: quoted}
    else:
        triple_context = self.__tripleContexts[triple] = {ctx: quoted, None: quoted}
    if not quoted:
        self.__contextTriples[None].add(triple)
    if ctx not in self.__contextTriples:
        self.__contextTriples[ctx] = set()
    self.__contextTriples[ctx].add(triple)
    if self.__defaultContexts is None:
        self.__defaultContexts = triple_context
    if triple_context == self.__defaultContexts:
        del self.__tripleContexts[triple]