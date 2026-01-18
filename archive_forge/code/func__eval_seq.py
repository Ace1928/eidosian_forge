from __future__ import annotations
import warnings
from functools import total_ordering
from typing import (
from rdflib.term import Node, URIRef
def _eval_seq(paths: List[Union[Path, URIRef]], subj: Optional[_SubjectType], obj: Optional[_ObjectType]) -> Generator[Tuple[_SubjectType, _ObjectType], None, None]:
    if paths[1:]:
        for s, o in eval_path(graph, (subj, paths[0], None)):
            for r in _eval_seq(paths[1:], o, obj):
                yield (s, r[1])
    else:
        for s, o in eval_path(graph, (subj, paths[0], obj)):
            yield (s, o)