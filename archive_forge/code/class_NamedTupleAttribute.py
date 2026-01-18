from .. import types, utils, errors
import operator
from .templates import (AttributeTemplate, ConcreteTemplate, AbstractTemplate,
from .builtins import normalize_1d_index
@infer_getattr
class NamedTupleAttribute(AttributeTemplate):
    key = types.BaseNamedTuple

    def resolve___class__(self, tup):
        return types.NamedTupleClass(tup.instance_class)

    def generic_resolve(self, tup, attr):
        try:
            index = tup.fields.index(attr)
        except ValueError:
            return
        return tup[index]