from xml.etree import ElementTree
from Bio.Phylo import PhyloXML as PX
def _ns(tag, namespace=NAMESPACES['phy']):
    """Format an XML tag with the given namespace (PRIVATE)."""
    return f'{{{namespace}}}{tag}'