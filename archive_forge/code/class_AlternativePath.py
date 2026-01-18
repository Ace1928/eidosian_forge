from __future__ import annotations
import warnings
from functools import total_ordering
from typing import (
from rdflib.term import Node, URIRef
class AlternativePath(Path):

    def __init__(self, *args: Union[Path, URIRef]):
        self.args: List[Union[Path, URIRef]] = []
        for a in args:
            if isinstance(a, AlternativePath):
                self.args += a.args
            else:
                self.args.append(a)

    def eval(self, graph: 'Graph', subj: Optional['_SubjectType']=None, obj: Optional['_ObjectType']=None) -> Generator[Tuple[_SubjectType, _ObjectType], None, None]:
        for x in self.args:
            for y in eval_path(graph, (subj, x, obj)):
                yield y

    def __repr__(self) -> str:
        return 'Path(%s)' % ' | '.join((str(x) for x in self.args))

    def n3(self, namespace_manager: Optional['NamespaceManager']=None) -> str:
        return '|'.join((_n3(a, namespace_manager) for a in self.args))