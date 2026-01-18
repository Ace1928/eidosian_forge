import logging
import xml.etree.ElementTree as xml_etree  # noqa: N813
from io import BytesIO
from typing import (
from xml.dom import XML_NAMESPACE
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesNSImpl
from rdflib.query import Result, ResultException, ResultParser, ResultSerializer
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
class SPARQLXMLWriter:
    """
    Python saxutils-based SPARQL XML Writer
    """

    def __init__(self, output: IO, encoding: str='utf-8'):
        writer = XMLGenerator(output, encoding)
        writer.startDocument()
        writer.startPrefixMapping('', SPARQL_XML_NAMESPACE)
        writer.startPrefixMapping('xml', XML_NAMESPACE)
        writer.startElementNS((SPARQL_XML_NAMESPACE, 'sparql'), 'sparql', AttributesNSImpl({}, {}))
        self.writer = writer
        self._output = output
        self._encoding = encoding
        self._results = False

    def write_header(self, allvarsL: Sequence[Variable]) -> None:
        self.writer.startElementNS((SPARQL_XML_NAMESPACE, 'head'), 'head', AttributesNSImpl({}, {}))
        for i in range(0, len(allvarsL)):
            attr_vals = {(None, 'name'): str(allvarsL[i])}
            attr_qnames = {(None, 'name'): 'name'}
            self.writer.startElementNS((SPARQL_XML_NAMESPACE, 'variable'), 'variable', AttributesNSImpl(attr_vals, attr_qnames))
            self.writer.endElementNS((SPARQL_XML_NAMESPACE, 'variable'), 'variable')
        self.writer.endElementNS((SPARQL_XML_NAMESPACE, 'head'), 'head')

    def write_ask(self, val: bool) -> None:
        self.writer.startElementNS((SPARQL_XML_NAMESPACE, 'boolean'), 'boolean', AttributesNSImpl({}, {}))
        self.writer.characters(str(val).lower())
        self.writer.endElementNS((SPARQL_XML_NAMESPACE, 'boolean'), 'boolean')

    def write_results_header(self) -> None:
        self.writer.startElementNS((SPARQL_XML_NAMESPACE, 'results'), 'results', AttributesNSImpl({}, {}))
        self._results = True

    def write_start_result(self) -> None:
        self.writer.startElementNS((SPARQL_XML_NAMESPACE, 'result'), 'result', AttributesNSImpl({}, {}))
        self._resultStarted = True

    def write_end_result(self) -> None:
        assert self._resultStarted
        self.writer.endElementNS((SPARQL_XML_NAMESPACE, 'result'), 'result')
        self._resultStarted = False

    def write_binding(self, name: Variable, val: Identifier) -> None:
        assert self._resultStarted
        attr_vals: Dict[Tuple[Optional[str], str], str] = {(None, 'name'): str(name)}
        attr_qnames: Dict[Tuple[Optional[str], str], str] = {(None, 'name'): 'name'}
        self.writer.startElementNS((SPARQL_XML_NAMESPACE, 'binding'), 'binding', AttributesNSImpl(attr_vals, attr_qnames))
        if isinstance(val, URIRef):
            self.writer.startElementNS((SPARQL_XML_NAMESPACE, 'uri'), 'uri', AttributesNSImpl({}, {}))
            self.writer.characters(val)
            self.writer.endElementNS((SPARQL_XML_NAMESPACE, 'uri'), 'uri')
        elif isinstance(val, BNode):
            self.writer.startElementNS((SPARQL_XML_NAMESPACE, 'bnode'), 'bnode', AttributesNSImpl({}, {}))
            self.writer.characters(val)
            self.writer.endElementNS((SPARQL_XML_NAMESPACE, 'bnode'), 'bnode')
        elif isinstance(val, Literal):
            attr_vals = {}
            attr_qnames = {}
            if val.language:
                attr_vals[XML_NAMESPACE, 'lang'] = val.language
                attr_qnames[XML_NAMESPACE, 'lang'] = 'xml:lang'
            elif val.datatype:
                attr_vals[None, 'datatype'] = val.datatype
                attr_qnames[None, 'datatype'] = 'datatype'
            self.writer.startElementNS((SPARQL_XML_NAMESPACE, 'literal'), 'literal', AttributesNSImpl(attr_vals, attr_qnames))
            self.writer.characters(val)
            self.writer.endElementNS((SPARQL_XML_NAMESPACE, 'literal'), 'literal')
        else:
            raise Exception('Unsupported RDF term: %s' % val)
        self.writer.endElementNS((SPARQL_XML_NAMESPACE, 'binding'), 'binding')

    def close(self) -> None:
        if self._results:
            self.writer.endElementNS((SPARQL_XML_NAMESPACE, 'results'), 'results')
        self.writer.endElementNS((SPARQL_XML_NAMESPACE, 'sparql'), 'sparql')
        self.writer.endDocument()