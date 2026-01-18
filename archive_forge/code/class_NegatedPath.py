from __future__ import annotations
import warnings
from functools import total_ordering
from typing import (
from rdflib.term import Node, URIRef
class NegatedPath(Path):

    def __init__(self, arg: Union[AlternativePath, InvPath, URIRef]):
        self.args: List[Union[URIRef, Path]]
        if isinstance(arg, (URIRef, InvPath)):
            self.args = [arg]
        elif isinstance(arg, AlternativePath):
            self.args = arg.args
        else:
            raise Exception('Can only negate URIRefs, InvPaths or ' + 'AlternativePaths, not: %s' % (arg,))

    def eval(self, graph, subj=None, obj=None):
        for s, p, o in graph.triples((subj, None, obj)):
            for a in self.args:
                if isinstance(a, URIRef):
                    if p == a:
                        break
                elif isinstance(a, InvPath):
                    if (o, a.arg, s) in graph:
                        break
                else:
                    raise Exception('Invalid path in NegatedPath: %s' % a)
            else:
                yield (s, o)

    def __repr__(self) -> str:
        return 'Path(! %s)' % ','.join((str(x) for x in self.args))

    def n3(self, namespace_manager: Optional['NamespaceManager']=None) -> str:
        return '!(%s)' % '|'.join((_n3(arg, namespace_manager) for arg in self.args))