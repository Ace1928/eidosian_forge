import collections
from llvmlite.ir import context, values, types, _utils
def _fix_metadata_operands(self, operands):
    fixed_ops = []
    for op in operands:
        if op is None:
            op = types.MetaDataType()(None)
        elif isinstance(op, str):
            op = values.MetaDataString(self, op)
        elif isinstance(op, (list, tuple)):
            op = self.add_metadata(op)
        fixed_ops.append(op)
    return fixed_ops