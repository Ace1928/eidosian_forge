import gast
from tensorflow.python.autograph.pyct import errors
class UnsupportedFeaturesChecker(gast.NodeVisitor):
    """Quick check for Python features we know we don't support.

  Any features detected will cause AutoGraph to not compile a function.
  """

    def visit_Attribute(self, node):
        if node.attr is not None and node.attr.startswith('__') and (not node.attr.endswith('__')):
            raise errors.UnsupportedLanguageElementError('mangled names are not yet supported')
        self.generic_visit(node)

    def visit_For(self, node):
        if node.orelse:
            raise errors.UnsupportedLanguageElementError('for/else statement not yet supported')
        self.generic_visit(node)

    def visit_While(self, node):
        if node.orelse:
            raise errors.UnsupportedLanguageElementError('while/else statement not yet supported')
        self.generic_visit(node)

    def visit_Yield(self, node):
        raise errors.UnsupportedLanguageElementError('generators are not supported')

    def visit_YieldFrom(self, node):
        raise errors.UnsupportedLanguageElementError('generators are not supported')