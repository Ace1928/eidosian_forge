import sys
import builtins as bltns
from types import MappingProxyType, DynamicClassAttribute
from operator import or_ as _or_
from functools import reduce
@classmethod
def _get_mixins_(mcls, class_name, bases):
    """
        Returns the type for creating enum members, and the first inherited
        enum class.

        bases: the tuple of bases that was given to __new__
        """
    if not bases:
        return (object, Enum)
    mcls._check_for_existing_members_(class_name, bases)
    first_enum = bases[-1]
    if not isinstance(first_enum, EnumType):
        raise TypeError('new enumerations should be created as `EnumName([mixin_type, ...] [data_type,] enum_type)`')
    member_type = mcls._find_data_type_(class_name, bases) or object
    return (member_type, first_enum)