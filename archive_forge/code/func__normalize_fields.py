import collections
import warnings
from functools import cached_property
from llvmlite import ir
from .abstract import DTypeSpec, IteratorType, MutableSequence, Number, Type
from .common import Buffer, Opaque, SimpleIteratorType
from numba.core.typeconv import Conversion
from numba.core import utils
from .misc import UnicodeType
from .containers import Bytes
import numpy as np
@classmethod
def _normalize_fields(cls, fields):
    """
        fields:
            [name: str,
             value: {
                 type: Type,
                 offset: int,
                 [ alignment: int ],
                 [ title : str],
             }]
        """
    res = []
    for name, infos in sorted(fields, key=lambda x: (x[1]['offset'], x[0])):
        fd = _RecordField(type=infos['type'], offset=infos['offset'], alignment=infos.get('alignment'), title=infos.get('title'))
        res.append((name, fd))
    return res