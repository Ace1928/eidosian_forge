from __future__ import absolute_import
from . import Nodes
from . import ExprNodes
from .Nodes import Node
from .ExprNodes import AtomicExprNode
from .PyrexTypes import c_ptr_type, c_bint_type
class TempsBlockNode(Node):
    """
    Creates a block which allocates temporary variables.
    This is used by transforms to output constructs that need
    to make use of a temporary variable. Simply pass the types
    of the needed temporaries to the constructor.

    The variables can be referred to using a TempRefNode
    (which can be constructed by calling get_ref_node).
    """
    child_attrs = ['body']

    def generate_execution_code(self, code):
        for handle in self.temps:
            handle.temp = code.funcstate.allocate_temp(handle.type, manage_ref=handle.needs_cleanup)
        self.body.generate_execution_code(code)
        for handle in self.temps:
            if handle.needs_cleanup:
                if handle.needs_xdecref:
                    code.put_xdecref_clear(handle.temp, handle.type)
                else:
                    code.put_decref_clear(handle.temp, handle.type)
            code.funcstate.release_temp(handle.temp)

    def analyse_declarations(self, env):
        self.body.analyse_declarations(env)

    def analyse_expressions(self, env):
        self.body = self.body.analyse_expressions(env)
        return self

    def generate_function_definitions(self, env, code):
        self.body.generate_function_definitions(env, code)

    def annotate(self, code):
        self.body.annotate(code)