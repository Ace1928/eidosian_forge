from xml.etree import ElementTree
from Bio.Phylo import PhyloXML as PX
def _handle_complex(tag, attribs, subnodes, has_text=False):
    """Handle to serialize nodes with subnodes (PRIVATE)."""

    def wrapped(self, obj):
        """Wrap nodes and subnodes as elements."""
        elem = ElementTree.Element(tag, _clean_attrib(obj, attribs))
        for subn in subnodes:
            if isinstance(subn, str):
                if getattr(obj, subn) is not None:
                    elem.append(getattr(self, subn)(getattr(obj, subn)))
            else:
                method, plural = subn
                for item in getattr(obj, plural):
                    elem.append(getattr(self, method)(item))
        if has_text:
            elem.text = _serialize(obj.value)
        return elem
    wrapped.__doc__ = f'Serialize a {tag} and its subnodes, in order.'
    return wrapped