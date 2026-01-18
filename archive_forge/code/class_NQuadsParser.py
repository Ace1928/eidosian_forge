from __future__ import annotations
from codecs import getreader
from typing import Any, MutableMapping, Optional
from rdflib.exceptions import ParserError as ParseError
from rdflib.graph import ConjunctiveGraph
from rdflib.parser import InputSource
from rdflib.plugins.parsers.ntriples import W3CNTriplesParser, r_tail, r_wspace
from rdflib.term import BNode
class NQuadsParser(W3CNTriplesParser):

    def parse(self, inputsource: InputSource, sink: ConjunctiveGraph, bnode_context: Optional[_BNodeContextType]=None, **kwargs: Any) -> ConjunctiveGraph:
        """
        Parse inputsource as an N-Quads file.

        :type inputsource: `rdflib.parser.InputSource`
        :param inputsource: the source of N-Quads-formatted data
        :type sink: `rdflib.graph.Graph`
        :param sink: where to send parsed triples
        :type bnode_context: `dict`, optional
        :param bnode_context: a dict mapping blank node identifiers to `~rdflib.term.BNode` instances.
                              See `.W3CNTriplesParser.parse`
        """
        assert sink.store.context_aware, 'NQuadsParser must be given a context aware store.'
        self.sink: ConjunctiveGraph = ConjunctiveGraph(store=sink.store, identifier=sink.identifier)
        source = inputsource.getCharacterStream()
        if not source:
            source = inputsource.getByteStream()
            source = getreader('utf-8')(source)
        if not hasattr(source, 'read'):
            raise ParseError('Item to parse must be a file-like object.')
        self.file = source
        self.buffer = ''
        while True:
            self.line = __line = self.readline()
            if self.line is None:
                break
            try:
                self.parseline(bnode_context)
            except ParseError as msg:
                raise ParseError('Invalid line (%s):\n%r' % (msg, __line))
        return self.sink

    def parseline(self, bnode_context: Optional[_BNodeContextType]=None) -> None:
        self.eat(r_wspace)
        if not self.line or self.line.startswith('#'):
            return
        subject = self.subject(bnode_context)
        self.eat(r_wspace)
        predicate = self.predicate()
        self.eat(r_wspace)
        obj = self.object(bnode_context)
        self.eat(r_wspace)
        context = self.uriref() or self.nodeid(bnode_context) or self.sink.identifier
        self.eat(r_tail)
        if self.line:
            raise ParseError('Trailing garbage')
        self.sink.get_context(context).add((subject, predicate, obj))