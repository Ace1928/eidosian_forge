from cupyx.jit import _cuda_types
from cupyx.jit import _cuda_typerules
from cupyx.jit import _internal_types
from cupy_backends.cuda.api import runtime as _runtime
class _ClassTemplate:

    def __init__(self, class_type):
        self._class_type = class_type
        self.__doc__ = self._class_type.__doc__

    def __getitem__(self, args):
        if isinstance(args, tuple):
            return self._class_type(*args)
        else:
            return self._class_type(args)