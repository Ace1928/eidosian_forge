from .. import types, utils, errors
import operator
from .templates import (AttributeTemplate, ConcreteTemplate, AbstractTemplate,
from .builtins import normalize_1d_index
@infer_global(operator.setitem)
class SetItemSequence(AbstractTemplate):

    def generic(self, args, kws):
        seq, idx, value = args
        if isinstance(seq, types.MutableSequence):
            idx = normalize_1d_index(idx)
            if isinstance(idx, types.SliceType):
                return signature(types.none, seq, idx, seq)
            elif isinstance(idx, types.Integer):
                if not self.context.can_convert(value, seq.dtype):
                    msg = 'invalid setitem with value of {} to element of {}'
                    raise errors.TypingError(msg.format(types.unliteral(value), seq.dtype))
                return signature(types.none, seq, idx, seq.dtype)