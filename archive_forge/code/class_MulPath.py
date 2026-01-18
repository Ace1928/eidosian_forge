from __future__ import annotations
import warnings
from functools import total_ordering
from typing import (
from rdflib.term import Node, URIRef
class MulPath(Path):

    def __init__(self, path: Union[Path, URIRef], mod: _MulPathMod):
        self.path = path
        self.mod = mod
        if mod == ZeroOrOne:
            self.zero = True
            self.more = False
        elif mod == ZeroOrMore:
            self.zero = True
            self.more = True
        elif mod == OneOrMore:
            self.zero = False
            self.more = True
        else:
            raise Exception('Unknown modifier %s' % mod)

    def eval(self, graph: 'Graph', subj: Optional['_SubjectType']=None, obj: Optional['_ObjectType']=None, first: bool=True) -> Generator[Tuple[_SubjectType, _ObjectType], None, None]:
        if self.zero and first:
            if subj and obj:
                if subj == obj:
                    yield (subj, obj)
            elif subj:
                yield (subj, subj)
            elif obj:
                yield (obj, obj)

        def _fwd(subj: Optional[_SubjectType]=None, obj: Optional[_ObjectType]=None, seen: Optional[Set[_SubjectType]]=None) -> Generator[Tuple[_SubjectType, _ObjectType], None, None]:
            seen.add(subj)
            for s, o in eval_path(graph, (subj, self.path, None)):
                if not obj or o == obj:
                    yield (s, o)
                if self.more:
                    if o in seen:
                        continue
                    for s2, o2 in _fwd(o, obj, seen):
                        yield (s, o2)

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
        done = set()
        if subj:
            for x in _fwd(subj, obj, set()):
                if x not in done:
                    done.add(x)
                    yield x
        elif obj:
            for x in _bwd(subj, obj, set()):
                if x not in done:
                    done.add(x)
                    yield x
        else:
            for x in _all_fwd_paths():
                if x not in done:
                    done.add(x)
                    yield x

    def __repr__(self) -> str:
        return 'Path(%s%s)' % (self.path, self.mod)

    def n3(self, namespace_manager: Optional['NamespaceManager']=None) -> str:
        return '%s%s' % (_n3(self.path, namespace_manager), self.mod)