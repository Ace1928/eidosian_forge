import re
from mako import ast
from mako import exceptions
from mako import filters
from mako import util
def accept_visitor(self, visitor):

    def traverse(node):
        for n in node.get_children():
            n.accept_visitor(visitor)
    method = getattr(visitor, 'visit' + self.__class__.__name__, traverse)
    method(self)