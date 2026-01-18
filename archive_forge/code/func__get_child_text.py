from xml.etree import ElementTree
from Bio.Phylo import PhyloXML as PX
def _get_child_text(parent, tag, construct=str):
    """Find a child node by tag; pass its text through a constructor (PRIVATE).

    Returns None if no matching child is found.
    """
    child = parent.find(_ns(tag))
    if child is not None and child.text:
        return construct(child.text)