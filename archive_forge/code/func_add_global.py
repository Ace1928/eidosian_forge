import collections
from llvmlite.ir import context, values, types, _utils
def add_global(self, globalvalue):
    """
        Add a new global value.
        """
    assert globalvalue.name not in self.globals
    self.globals[globalvalue.name] = globalvalue