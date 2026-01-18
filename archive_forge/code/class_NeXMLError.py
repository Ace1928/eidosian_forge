from io import StringIO
from xml.dom import minidom
from xml.etree import ElementTree
from Bio.Phylo import NeXML
from ._cdao_owl import cdao_elements, cdao_namespaces, resolve_uri
class NeXMLError(Exception):
    """Exception raised when NeXML object construction cannot continue."""