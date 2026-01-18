from __future__ import annotations
from typing import (
from rdflib.store import Store
from rdflib.util import _coalesce
def __remove_triple_context(self, triple: '_TripleType', ctx):
    """remove the context from the triple"""
    ctxs = self.__tripleContexts.get(triple, self.__defaultContexts).copy()
    del ctxs[ctx]
    if ctxs == self.__defaultContexts:
        del self.__tripleContexts[triple]
    else:
        self.__tripleContexts[triple] = ctxs
    self.__contextTriples[ctx].remove(triple)