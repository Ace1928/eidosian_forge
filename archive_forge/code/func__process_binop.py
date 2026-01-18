import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import templates
def _process_binop(self, op, left, right):
    overload = self._overload_of(op)
    if overload is None:
        return self._as_binary_operation(op, left, right)
    return self._as_binary_function(overload, left, right)