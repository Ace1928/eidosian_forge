from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def GetNodeText(node):
    """Returns the node text after stripping whitespace."""
    return node.text.strip() if node.text else ''