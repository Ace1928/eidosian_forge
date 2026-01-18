from copy import copy
from . import ast
from .visitor_meta import QUERY_DOCUMENT_KEYS, VisitorMeta
class Visitor(metaclass=VisitorMeta):
    __slots__ = ()

    def enter(self, node, key, parent, path, ancestors):
        method = self._get_enter_handler(type(node))
        if method:
            return method(self, node, key, parent, path, ancestors)

    def leave(self, node, key, parent, path, ancestors):
        method = self._get_leave_handler(type(node))
        if method:
            return method(self, node, key, parent, path, ancestors)