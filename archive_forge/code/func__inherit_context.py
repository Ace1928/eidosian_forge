import string
from xml.dom import Node
def _inherit_context(self, node):
    """_inherit_context(self, node) -> list
        Scan ancestors of attribute and namespace context.  Used only
        for single element node canonicalization, not for subset
        canonicalization."""
    xmlattrs = filter(_IN_XML_NS, _attrs(node))
    inherited, parent = ([], node.parentNode)
    while parent and parent.nodeType == Node.ELEMENT_NODE:
        for a in filter(_IN_XML_NS, _attrs(parent)):
            n = a.localName
            if n not in xmlattrs:
                xmlattrs.append(n)
                inherited.append(a)
        parent = parent.parentNode
    return inherited