import logging
import xml.etree.ElementTree as xml_etree  # noqa: N813
from io import BytesIO
from typing import (
from xml.dom import XML_NAMESPACE
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesNSImpl
from rdflib.query import Result, ResultException, ResultParser, ResultSerializer
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
class XMLResultSerializer(ResultSerializer):

    def __init__(self, result: Result):
        ResultSerializer.__init__(self, result)

    def serialize(self, stream: IO, encoding: str='utf-8', **kwargs: Any) -> None:
        writer = SPARQLXMLWriter(stream, encoding)
        if self.result.type == 'ASK':
            writer.write_header([])
            writer.write_ask(self.result.askAnswer)
        else:
            writer.write_header(self.result.vars)
            writer.write_results_header()
            for b in self.result.bindings:
                writer.write_start_result()
                for key, val in b.items():
                    writer.write_binding(key, val)
                writer.write_end_result()
        writer.close()