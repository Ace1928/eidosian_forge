import sys
import builtins as bltns
from types import MappingProxyType, DynamicClassAttribute
from operator import or_ as _or_
from functools import reduce
@classmethod
def _find_new_(mcls, classdict, member_type, first_enum):
    """
        Returns the __new__ to be used for creating the enum members.

        classdict: the class dictionary given to __new__
        member_type: the data type whose __new__ will be used by default
        first_enum: enumeration to check for an overriding __new__
        """
    __new__ = classdict.get('__new__', None)
    save_new = first_enum is not None and __new__ is not None
    if __new__ is None:
        for method in ('__new_member__', '__new__'):
            for possible in (member_type, first_enum):
                target = getattr(possible, method, None)
                if target not in {None, None.__new__, object.__new__, Enum.__new__}:
                    __new__ = target
                    break
            if __new__ is not None:
                break
        else:
            __new__ = object.__new__
    if first_enum is None or __new__ in (Enum.__new__, object.__new__):
        use_args = False
    else:
        use_args = True
    return (__new__, save_new, use_args)