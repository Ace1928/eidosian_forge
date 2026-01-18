from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def GetNodes(node, match_tag):
    """Gets all children of a node with the desired tag."""
    return (child for child in list(node) if GetTag(child) == match_tag)