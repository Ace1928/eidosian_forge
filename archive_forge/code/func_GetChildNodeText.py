from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def GetChildNodeText(node, child_tag, default=''):
    """Finds child xml node with desired tag and returns its text."""
    for child in list(node):
        if GetTag(child) == child_tag:
            return GetNodeText(child) or default
    return default