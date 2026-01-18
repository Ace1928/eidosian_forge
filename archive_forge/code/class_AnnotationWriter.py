from __future__ import absolute_import, print_function
from .Visitor import CythonTransform
from .StringEncoding import EncodedString
from . import Options
from . import PyrexTypes
from ..CodeWriter import ExpressionWriter
from .Errors import warning
class AnnotationWriter(ExpressionWriter):
    """
    A Cython code writer for Python expressions in argument/variable annotations.
    """

    def __init__(self, description=None):
        """description is optional. If specified it is used in
        warning messages for the nodes that don't convert to string properly.
        If not specified then no messages are generated.
        """
        ExpressionWriter.__init__(self)
        self.description = description
        self.incomplete = False

    def visit_Node(self, node):
        self.put(u'<???>')
        self.incomplete = True
        if self.description:
            warning(node.pos, 'Failed to convert code to string representation in {0}'.format(self.description), level=1)

    def visit_LambdaNode(self, node):
        self.put('<lambda>')
        self.incomplete = True
        if self.description:
            warning(node.pos, 'Failed to convert lambda to string representation in {0}'.format(self.description), level=1)

    def visit_UnicodeNode(self, node):
        self.emit_string(node, '')

    def visit_AnnotationNode(self, node):
        self.put(node.string.unicode_value)