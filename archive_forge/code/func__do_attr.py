import string
from xml.dom import Node
def _do_attr(self, n, value):
    """'_do_attr(self, node) -> None
        Process an attribute."""
    W = self.write
    W(' ')
    W(n)
    W('="')
    s = string.replace(value, '&', '&amp;')
    s = string.replace(s, '<', '&lt;')
    s = string.replace(s, '"', '&quot;')
    s = string.replace(s, '\t', '&#x9')
    s = string.replace(s, '\n', '&#xA')
    s = string.replace(s, '\r', '&#xD')
    W(s)
    W('"')