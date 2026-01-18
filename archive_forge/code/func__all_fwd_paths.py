from __future__ import annotations
import warnings
from functools import total_ordering
from typing import (
from rdflib.term import Node, URIRef
def _all_fwd_paths() -> Generator[Tuple[_SubjectType, _ObjectType], None, None]:
    if self.zero:
        seen1 = set()
        for s, o in graph.subject_objects(None):
            if s not in seen1:
                seen1.add(s)
                yield (s, s)
            if o not in seen1:
                seen1.add(o)
                yield (o, o)
    seen = set()
    for s, o in eval_path(graph, (None, self.path, None)):
        if not self.more:
            yield (s, o)
        elif s not in seen:
            seen.add(s)
            f = list(_fwd(s, None, set()))
            for s1, o1 in f:
                assert s1 == s
                yield (s1, o1)