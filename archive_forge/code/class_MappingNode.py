from __future__ import print_function
import sys
from .compat import string_types
class MappingNode(CollectionNode):
    __slots__ = ('merge',)
    id = 'mapping'

    def __init__(self, tag, value, start_mark=None, end_mark=None, flow_style=None, comment=None, anchor=None):
        CollectionNode.__init__(self, tag, value, start_mark, end_mark, flow_style, comment, anchor)
        self.merge = None