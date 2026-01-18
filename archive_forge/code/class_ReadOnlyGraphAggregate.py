from __future__ import annotations
import logging
import pathlib
import random
from io import BytesIO
from typing import (
from urllib.parse import urlparse
from urllib.request import url2pathname
import rdflib.exceptions as exceptions
import rdflib.namespace as namespace  # noqa: F401 # This is here because it is used in a docstring.
import rdflib.plugin as plugin
import rdflib.query as query
import rdflib.util  # avoid circular dependency
from rdflib.collection import Collection
from rdflib.exceptions import ParserError
from rdflib.namespace import RDF, Namespace, NamespaceManager
from rdflib.parser import InputSource, Parser, create_input_source
from rdflib.paths import Path
from rdflib.resource import Resource
from rdflib.serializer import Serializer
from rdflib.store import Store
from rdflib.term import (
class ReadOnlyGraphAggregate(ConjunctiveGraph):
    """Utility class for treating a set of graphs as a single graph

    Only read operations are supported (hence the name). Essentially a
    ConjunctiveGraph over an explicit subset of the entire store.
    """

    def __init__(self, graphs: List[Graph], store: Union[str, Store]='default'):
        if store is not None:
            super(ReadOnlyGraphAggregate, self).__init__(store)
            Graph.__init__(self, store)
            self.__namespace_manager = None
        assert isinstance(graphs, list) and graphs and [g for g in graphs if isinstance(g, Graph)], 'graphs argument must be a list of Graphs!!'
        self.graphs = graphs

    def __repr__(self) -> str:
        return '<ReadOnlyGraphAggregate: %s graphs>' % len(self.graphs)

    def destroy(self, configuration: str) -> NoReturn:
        raise ModificationException()

    def commit(self) -> NoReturn:
        raise ModificationException()

    def rollback(self) -> NoReturn:
        raise ModificationException()

    def open(self, configuration: str, create: bool=False) -> None:
        for graph in self.graphs:
            graph.open(self, configuration, create)

    def close(self) -> None:
        for graph in self.graphs:
            graph.close()

    def add(self, triple: _TripleOrOptionalQuadType) -> NoReturn:
        raise ModificationException()

    def addN(self, quads: Iterable['_QuadType']) -> NoReturn:
        raise ModificationException()

    def remove(self, triple: _TripleOrOptionalQuadType) -> NoReturn:
        raise ModificationException()

    @overload
    def triples(self, triple: '_TriplePatternType') -> Generator['_TripleType', None, None]:
        ...

    @overload
    def triples(self, triple: '_TriplePathPatternType') -> Generator['_TriplePathType', None, None]:
        ...

    @overload
    def triples(self, triple: '_TripleSelectorType') -> Generator['_TripleOrTriplePathType', None, None]:
        ...

    def triples(self, triple: '_TripleSelectorType') -> Generator['_TripleOrTriplePathType', None, None]:
        s, p, o = triple
        for graph in self.graphs:
            if isinstance(p, Path):
                for s, o in p.eval(self, s, o):
                    yield (s, p, o)
            else:
                for s1, p1, o1 in graph.triples((s, p, o)):
                    yield (s1, p1, o1)

    def __contains__(self, triple_or_quad: _TripleOrQuadSelectorType) -> bool:
        context = None
        if len(triple_or_quad) == 4:
            context = triple_or_quad[3]
        for graph in self.graphs:
            if context is None or graph.identifier == context.identifier:
                if triple_or_quad[:3] in graph:
                    return True
        return False

    def quads(self, triple_or_quad: _TripleOrQuadSelectorType) -> Generator[Tuple['_SubjectType', Union[Path, '_PredicateType'], '_ObjectType', '_ContextType'], None, None]:
        """Iterate over all the quads in the entire aggregate graph"""
        c = None
        if len(triple_or_quad) == 4:
            s, p, o, c = triple_or_quad
        else:
            s, p, o = triple_or_quad
        if c is not None:
            for graph in [g for g in self.graphs if g == c]:
                for s1, p1, o1 in graph.triples((s, p, o)):
                    yield (s1, p1, o1, graph)
        else:
            for graph in self.graphs:
                for s1, p1, o1 in graph.triples((s, p, o)):
                    yield (s1, p1, o1, graph)

    def __len__(self) -> int:
        return sum((len(g) for g in self.graphs))

    def __hash__(self) -> NoReturn:
        raise UnSupportedAggregateOperation()

    def __cmp__(self, other) -> int:
        if other is None:
            return -1
        elif isinstance(other, Graph):
            return -1
        elif isinstance(other, ReadOnlyGraphAggregate):
            return (self.graphs > other.graphs) - (self.graphs < other.graphs)
        else:
            return -1

    def __iadd__(self: '_GraphT', other: Iterable['_TripleType']) -> NoReturn:
        raise ModificationException()

    def __isub__(self: '_GraphT', other: Iterable['_TripleType']) -> NoReturn:
        raise ModificationException()

    def triples_choices(self, triple: Union[Tuple[List['_SubjectType'], '_PredicateType', '_ObjectType'], Tuple['_SubjectType', List['_PredicateType'], '_ObjectType'], Tuple['_SubjectType', '_PredicateType', List['_ObjectType']]], context: Optional['_ContextType']=None) -> Generator[_TripleType, None, None]:
        subject, predicate, object_ = triple
        for graph in self.graphs:
            choices = graph.triples_choices((subject, predicate, object_))
            for s, p, o in choices:
                yield (s, p, o)

    def qname(self, uri: str) -> str:
        if hasattr(self, 'namespace_manager') and self.namespace_manager:
            return self.namespace_manager.qname(uri)
        raise UnSupportedAggregateOperation()

    def compute_qname(self, uri: str, generate: bool=True) -> Tuple[str, URIRef, str]:
        if hasattr(self, 'namespace_manager') and self.namespace_manager:
            return self.namespace_manager.compute_qname(uri, generate)
        raise UnSupportedAggregateOperation()

    def bind(self, prefix: Optional[str], namespace: Any, override: bool=True) -> NoReturn:
        raise UnSupportedAggregateOperation()

    def namespaces(self) -> Generator[Tuple[str, URIRef], None, None]:
        if hasattr(self, 'namespace_manager'):
            for prefix, namespace in self.namespace_manager.namespaces():
                yield (prefix, namespace)
        else:
            for graph in self.graphs:
                for prefix, namespace in graph.namespaces():
                    yield (prefix, namespace)

    def absolutize(self, uri: str, defrag: int=1) -> NoReturn:
        raise UnSupportedAggregateOperation()

    def parse(self, source: Optional[Union[IO[bytes], TextIO, InputSource, str, bytes, pathlib.PurePath]], publicID: Optional[str]=None, format: Optional[str]=None, **args: Any) -> NoReturn:
        raise ModificationException()

    def n3(self) -> NoReturn:
        raise UnSupportedAggregateOperation()

    def __reduce__(self) -> NoReturn:
        raise UnSupportedAggregateOperation()