from __future__ import annotations
import warnings
from functools import total_ordering
from typing import (
from rdflib.term import Node, URIRef
def _bwd(subj: Optional[_SubjectType]=None, obj: Optional[_ObjectType]=None, seen: Optional[Set[_ObjectType]]=None) -> Generator[Tuple[_SubjectType, _ObjectType], None, None]:
    seen.add(obj)
    for s, o in eval_path(graph, (None, self.path, obj)):
        if not subj or subj == s:
            yield (s, o)
        if self.more:
            if s in seen:
                continue
            for s2, o2 in _bwd(None, s, seen):
                yield (s2, o)