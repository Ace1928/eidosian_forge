import random
from jmespath import lexer
from jmespath.compat import with_repr_method
from jmespath import ast
from jmespath import exceptions
from jmespath import visitor
@with_repr_method
class ParsedResult(object):

    def __init__(self, expression, parsed):
        self.expression = expression
        self.parsed = parsed

    def search(self, value, options=None):
        interpreter = visitor.TreeInterpreter(options)
        result = interpreter.visit(self.parsed, value)
        return result

    def _render_dot_file(self):
        """Render the parsed AST as a dot file.

        Note that this is marked as an internal method because
        the AST is an implementation detail and is subject
        to change.  This method can be used to help troubleshoot
        or for development purposes, but is not considered part
        of the public supported API.  Use at your own risk.

        """
        renderer = visitor.GraphvizVisitor()
        contents = renderer.visit(self.parsed)
        return contents

    def __repr__(self):
        return repr(self.parsed)