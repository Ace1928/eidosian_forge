import numpy as np
import operator
from collections import namedtuple
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.core.typing import collections
from numba.core.errors import (TypingError, RequireLiteralValue, NumbaTypeError,
from numba.core.cgutils import is_nonelike
@infer_global(operator.getitem)
class StaticGetItemLiteralRecord(AbstractTemplate):

    def generic(self, args, kws):
        record, idx = args
        if isinstance(record, types.Record):
            if isinstance(idx, types.StringLiteral):
                if idx.literal_value not in record.fields:
                    msg = f"Field '{idx.literal_value}' was not found in record with fields {tuple(record.fields.keys())}"
                    raise NumbaKeyError(msg)
                ret = record.typeof(idx.literal_value)
                assert ret
                return signature(ret, *args)
            elif isinstance(idx, types.IntegerLiteral):
                if idx.literal_value >= len(record.fields):
                    msg = f'Requested index {idx.literal_value} is out of range'
                    raise NumbaIndexError(msg)
                field_names = list(record.fields)
                ret = record.typeof(field_names[idx.literal_value])
                assert ret
                return signature(ret, *args)