from .. import types, utils, errors
import operator
from .templates import (AttributeTemplate, ConcreteTemplate, AbstractTemplate,
from .builtins import normalize_1d_index
@infer_global(len)
class ContainerLen(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        val, = args
        if isinstance(val, types.Container):
            return signature(types.intp, val)