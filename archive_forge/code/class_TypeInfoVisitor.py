from copy import copy
from . import ast
from .visitor_meta import QUERY_DOCUMENT_KEYS, VisitorMeta
class TypeInfoVisitor(Visitor):
    __slots__ = ('visitor', 'type_info')

    def __init__(self, type_info, visitor):
        self.type_info = type_info
        self.visitor = visitor

    def enter(self, node, key, parent, path, ancestors):
        self.type_info.enter(node)
        result = self.visitor.enter(node, key, parent, path, ancestors)
        if result is not None:
            self.type_info.leave(node)
            if isinstance(result, ast.Node):
                self.type_info.enter(result)
        return result

    def leave(self, node, key, parent, path, ancestors):
        result = self.visitor.leave(node, key, parent, path, ancestors)
        self.type_info.leave(node)
        return result