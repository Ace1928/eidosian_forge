import operator
from numba.core import types
from numba.core.typing.templates import (AbstractTemplate, AttributeTemplate,
@infer_getattr
class EnumClassAttribute(AttributeTemplate):
    key = types.EnumClass

    def generic_resolve(self, ty, attr):
        """
        Resolve attributes of an enum class as enum members.
        """
        if attr in ty.instance_class.__members__:
            return ty.member_type