from rdflib import BNode
from rdflib import Literal, URIRef
from rdflib import RDF as ns_rdf
from rdflib.term import XSDToPython
from . import IncorrectBlankNodeUsage, IncorrectLiteral, err_no_blank_node
from .utils import has_one_of_attributes, return_XML
import re
def _putBackEntities(self, data):
    """Put 'back' entities for the '&','<', and '>' characters, to produce a proper XML string.
        Used by the XML Literal extraction.
        @param data: string to be converted
        @return: string with entities
        @rtype: string
        """
    return data.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')