from xml.etree import ElementTree
from Bio.Phylo import PhyloXML as PX
def _get_child_as(parent, tag, construct):
    """Find a child node by tag, and pass it through a constructor (PRIVATE).

    Returns None if no matching child is found.
    """
    child = parent.find(_ns(tag))
    if child is not None:
        return construct(child)