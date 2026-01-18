import itertools
import numpy as np
import operator
from numba.core import types, errors
from numba import prange
from numba.parfors.parfor import internal_prange
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cpython.builtins import get_type_min_value, get_type_max_value
from numba.core.extending import (
@infer_getattr
class TypeRefAttribute(AttributeTemplate):
    key = types.TypeRef

    def resolve___call__(self, classty):
        """
        Resolve a core number's constructor (e.g. calling int(...))

        Note:

        This is needed because of the limitation of the current type-system
        implementation.  Specifically, the lack of a higher-order type
        (i.e. passing the ``DictType`` vs ``DictType(key_type, value_type)``)
        """
        ty = classty.instance_type
        if isinstance(ty, type) and issubclass(ty, types.Type):

            class Redirect(object):

                def __init__(self, context):
                    self.context = context

                def __call__(self, *args, **kwargs):
                    result = self.context.resolve_function_type(ty, args, kwargs)
                    if hasattr(result, 'pysig'):
                        self.pysig = result.pysig
                    return result
            return types.Function(make_callable_template(key=ty, typer=Redirect(self.context)))