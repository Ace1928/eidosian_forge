from xml.etree import ElementTree
from Bio.Phylo import PhyloXML as PX
def _collapse_wspace(text):
    """Replace all spans of whitespace with a single space character (PRIVATE).

    Also remove leading and trailing whitespace. See "Collapse Whitespace
    Policy" in the phyloXML spec glossary:
    http://phyloxml.org/documentation/version_100/phyloxml.xsd.html#Glossary
    """
    if text is not None:
        return ' '.join(text.split())