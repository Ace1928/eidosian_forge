from xml.etree import ElementTree
from Bio.Phylo import PhyloXML as PX
def _clean_attrib(obj, attrs):
    """Create a dictionary from an object's specified, non-None attributes (PRIVATE)."""
    out = {}
    for key in attrs:
        val = getattr(obj, key)
        if val is not None:
            out[key] = _serialize(val)
    return out