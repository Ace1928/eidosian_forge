from .. import types, utils, errors
import operator
from .templates import (AttributeTemplate, ConcreteTemplate, AbstractTemplate,
from .builtins import normalize_1d_index
@infer_global(operator.truth)
class SequenceBool(AbstractTemplate):
    key = operator.truth

    def generic(self, args, kws):
        assert not kws
        val, = args
        if isinstance(val, types.Sequence):
            return signature(types.boolean, val)