from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def GetAttribute(node, attr):
    """Wrapper function to retrieve attributes from XML nodes."""
    return node.attrib.get(attr, '')