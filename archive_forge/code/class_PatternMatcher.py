import ast
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
class PatternMatcher(gast.NodeVisitor):
    """Matches a node against a pattern represented by a node."""

    def __init__(self, pattern):
        self.pattern = pattern
        self.pattern_stack = []
        self.matches = True

    def compare_and_visit(self, node, pattern):
        self.pattern_stack.append(self.pattern)
        self.pattern = pattern
        self.generic_visit(node)
        self.pattern = self.pattern_stack.pop()

    def no_match(self):
        self.matches = False
        return False

    def is_wildcard(self, p):
        if isinstance(p, (list, tuple)) and len(p) == 1:
            p, = p
        if isinstance(p, gast.Name) and p.id == '_':
            return True
        if p == '_':
            return True
        return False

    def generic_visit(self, node):
        if not self.matches:
            return
        pattern = self.pattern
        for f in node._fields:
            if f.startswith('__'):
                continue
            if not hasattr(node, f):
                if hasattr(pattern, f) and getattr(pattern, f):
                    return self.no_match()
                else:
                    continue
            if not hasattr(pattern, f):
                return self.no_match()
            v = getattr(node, f)
            p = getattr(pattern, f)
            if self.is_wildcard(p):
                continue
            if isinstance(v, (list, tuple)):
                if not isinstance(p, (list, tuple)) or len(v) != len(p):
                    return self.no_match()
                for v_item, p_item in zip(v, p):
                    self.compare_and_visit(v_item, p_item)
            elif isinstance(v, (gast.AST, ast.AST)):
                if not isinstance(v, type(p)) and (not isinstance(p, type(v))):
                    return self.no_match()
                self.compare_and_visit(v, p)
            elif v != p:
                return self.no_match()