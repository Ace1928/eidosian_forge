import operator
from numba.core import types
from numba.core.typing.templates import (AbstractTemplate, AttributeTemplate,
@infer
class EnumClassStaticGetItem(AbstractTemplate):
    key = 'static_getitem'

    def generic(self, args, kws):
        enum, idx = args
        if isinstance(enum, types.EnumClass) and idx in enum.instance_class.__members__:
            return signature(enum.member_type, *args)