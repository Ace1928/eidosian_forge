import collections
from llvmlite.ir import context, values, types, _utils
def add_debug_info(self, kind, operands, is_distinct=False):
    """
        Add debug information metadata to the module with the given
        *operands* (a dict of values with string keys) or return
        a previous equivalent metadata.  *kind* is a string of the
        debug information kind (e.g. "DICompileUnit").

        A DIValue instance is returned, it can then be associated to e.g.
        an instruction.
        """
    operands = tuple(sorted(self._fix_di_operands(operands.items())))
    key = (kind, operands, is_distinct)
    if key not in self._metadatacache:
        n = len(self.metadata)
        di = values.DIValue(self, is_distinct, kind, operands, name=str(n))
        self._metadatacache[key] = di
    else:
        di = self._metadatacache[key]
    return di