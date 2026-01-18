from __future__ import print_function
import sys
from .compat import string_types
class CollectionNode(Node):
    __slots__ = ('flow_style',)

    def __init__(self, tag, value, start_mark=None, end_mark=None, flow_style=None, comment=None, anchor=None):
        Node.__init__(self, tag, value, start_mark, end_mark, comment=comment)
        self.flow_style = flow_style
        self.anchor = anchor