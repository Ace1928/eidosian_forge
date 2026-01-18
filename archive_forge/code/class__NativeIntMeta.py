from typing import Any
import sys
from typing import _type_check  # type: ignore
class _NativeIntMeta(type):

    def __instancecheck__(cls, inst):
        return isinstance(inst, int)