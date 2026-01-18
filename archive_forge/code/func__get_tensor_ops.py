import inspect
import textwrap
import torch.jit
from torch.jit._builtins import _find_builtin
def _get_tensor_ops():

    def is_tensor_method(schema):
        if len(schema.arguments) == 0:
            return False
        self = schema.arguments[0]
        if self.name != 'self':
            return False
        if not self.type.isSubtypeOf(torch._C.TensorType.get()):
            return False
        return True
    methods = []
    for elem in dir(torch.Tensor):
        if not _hidden(elem):
            schemas = torch._C._jit_get_schemas_for_operator('aten::' + elem)
            for schema in schemas:
                if is_tensor_method(schema):
                    methods.append(_emit_schema('Tensor', elem, schema, arg_start=1))
    return ('Supported Tensor Methods', methods)