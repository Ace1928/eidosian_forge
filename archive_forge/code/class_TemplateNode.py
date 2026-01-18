import re
from mako import ast
from mako import exceptions
from mako import filters
from mako import util
class TemplateNode(Node):
    """a 'container' node that stores the overall collection of nodes."""

    def __init__(self, filename):
        super().__init__('', 0, 0, filename)
        self.nodes = []
        self.page_attributes = {}

    def get_children(self):
        return self.nodes

    def __repr__(self):
        return 'TemplateNode(%s, %r)' % (util.sorted_dict_repr(self.page_attributes), self.nodes)