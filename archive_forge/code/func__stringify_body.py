import collections
from llvmlite.ir import context, values, types, _utils
def _stringify_body(self):
    return '\n'.join(self._get_body_lines())