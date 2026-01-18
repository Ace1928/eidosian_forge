from __future__ import annotations
from typing import Any, MutableSequence
from rdflib.graph import ConjunctiveGraph, Graph
from rdflib.parser import InputSource, Parser
from .notation3 import RDFSink, SinkParser
class TrigParser(Parser):
    """
    An RDFLib parser for TriG

    """

    def __init__(self):
        pass

    def parse(self, source: InputSource, graph: Graph, encoding: str='utf-8') -> None:
        if encoding not in [None, 'utf-8']:
            raise Exception(('TriG files are always utf-8 encoded, ', 'I was passed: %s') % encoding)
        assert graph.store.context_aware, 'TriG Parser needs a context-aware store!'
        conj_graph = ConjunctiveGraph(store=graph.store, identifier=graph.identifier)
        conj_graph.default_context = graph
        conj_graph.namespace_manager = graph.namespace_manager
        sink = RDFSink(conj_graph)
        baseURI = conj_graph.absolutize(source.getPublicId() or source.getSystemId() or '')
        p = TrigSinkParser(sink, baseURI=baseURI, turtle=True)
        stream = source.getCharacterStream()
        if not stream:
            stream = source.getByteStream()
        p.loadStream(stream)
        for prefix, namespace in p._bindings.items():
            conj_graph.bind(prefix, namespace)