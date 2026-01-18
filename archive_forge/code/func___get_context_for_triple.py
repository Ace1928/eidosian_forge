from __future__ import annotations
from typing import (
from rdflib.store import Store
from rdflib.util import _coalesce
def __get_context_for_triple(self, triple: '_TripleType', skipQuoted: bool=False) -> Collection[Optional[str]]:
    """return a list of contexts (str) for the triple, skipping
        quoted contexts if skipQuoted==True"""
    ctxs = self.__tripleContexts.get(triple, self.__defaultContexts)
    if not skipQuoted:
        return ctxs.keys()
    return [ctx for ctx, quoted in ctxs.items() if not quoted]