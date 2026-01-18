from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def GetChild(node, tag):
    """Returns first child of node with tag."""
    for child in list(node):
        if GetTag(child) == tag:
            return child