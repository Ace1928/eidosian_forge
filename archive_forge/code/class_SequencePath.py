from __future__ import annotations
import warnings
from functools import total_ordering
from typing import (
from rdflib.term import Node, URIRef
class SequencePath(Path):

    def __init__(self, *args: Union[Path, URIRef]):
        self.args: List[Union[Path, URIRef]] = []
        for a in args:
            if isinstance(a, SequencePath):
                self.args += a.args
            else:
                self.args.append(a)

    def eval(self, graph: 'Graph', subj: Optional['_SubjectType']=None, obj: Optional['_ObjectType']=None) -> Generator[Tuple[_SubjectType, _ObjectType], None, None]:

        def _eval_seq(paths: List[Union[Path, URIRef]], subj: Optional[_SubjectType], obj: Optional[_ObjectType]) -> Generator[Tuple[_SubjectType, _ObjectType], None, None]:
            if paths[1:]:
                for s, o in eval_path(graph, (subj, paths[0], None)):
                    for r in _eval_seq(paths[1:], o, obj):
                        yield (s, r[1])
            else:
                for s, o in eval_path(graph, (subj, paths[0], obj)):
                    yield (s, o)

        def _eval_seq_bw(paths: List[Union[Path, URIRef]], subj: Optional[_SubjectType], obj: _ObjectType) -> Generator[Tuple[_SubjectType, _ObjectType], None, None]:
            if paths[:-1]:
                for s, o in eval_path(graph, (None, paths[-1], obj)):
                    for r in _eval_seq(paths[:-1], subj, s):
                        yield (r[0], o)
            else:
                for s, o in eval_path(graph, (subj, paths[0], obj)):
                    yield (s, o)
        if subj:
            return _eval_seq(self.args, subj, obj)
        elif obj:
            return _eval_seq_bw(self.args, subj, obj)
        else:
            return _eval_seq(self.args, subj, obj)

    def __repr__(self) -> str:
        return 'Path(%s)' % ' / '.join((str(x) for x in self.args))

    def n3(self, namespace_manager: Optional['NamespaceManager']=None) -> str:
        return '/'.join((_n3(a, namespace_manager) for a in self.args))