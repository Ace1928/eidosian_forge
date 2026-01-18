import logging
import xml.etree.ElementTree as xml_etree  # noqa: N813
from io import BytesIO
from typing import (
from xml.dom import XML_NAMESPACE
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesNSImpl
from rdflib.query import Result, ResultException, ResultParser, ResultSerializer
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
class XMLResult(Result):

    def __init__(self, source: IO, content_type: Optional[str]=None):
        parser_encoding: Optional[str] = None
        if hasattr(source, 'encoding'):
            if TYPE_CHECKING:
                assert isinstance(source, TextIO)
            parser_encoding = 'utf-8'
            source_str = source.read()
            source = BytesIO(source_str.encode(parser_encoding))
        elif TYPE_CHECKING:
            assert isinstance(source, BinaryIO)
        if FOUND_LXML:
            lxml_parser = lxml_etree.XMLParser(huge_tree=True, encoding=parser_encoding)
            tree = cast(xml_etree.ElementTree, lxml_etree.parse(source, parser=lxml_parser))
        else:
            xml_parser = xml_etree.XMLParser(encoding=parser_encoding)
            tree = xml_etree.parse(source, parser=xml_parser)
        boolean = tree.find(RESULTS_NS_ET + 'boolean')
        results = tree.find(RESULTS_NS_ET + 'results')
        if boolean is not None:
            type_ = 'ASK'
        elif results is not None:
            type_ = 'SELECT'
        else:
            raise ResultException('No RDF result-bindings or boolean answer found!')
        Result.__init__(self, type_)
        if type_ == 'SELECT':
            self.bindings = []
            for result in results:
                if result.tag != f'{RESULTS_NS_ET}result':
                    continue
                r = {}
                for binding in result:
                    if binding.tag != f'{RESULTS_NS_ET}binding':
                        continue
                    r[Variable(binding.get('name'))] = parseTerm(binding[0])
                self.bindings.append(r)
            self.vars = [Variable(x.get('name')) for x in tree.findall('./%shead/%svariable' % (RESULTS_NS_ET, RESULTS_NS_ET))]
        else:
            self.askAnswer = boolean.text.lower().strip() == 'true'