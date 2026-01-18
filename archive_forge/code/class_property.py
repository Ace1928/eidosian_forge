import sys
import builtins as bltns
from types import MappingProxyType, DynamicClassAttribute
from operator import or_ as _or_
from functools import reduce
class property(DynamicClassAttribute):
    """
    This is a descriptor, used to define attributes that act differently
    when accessed through an enum member and through an enum class.
    Instance access is the same as property(), but access to an attribute
    through the enum class will instead look in the class' _member_map_ for
    a corresponding enum member.
    """

    def __get__(self, instance, ownerclass=None):
        if instance is None:
            try:
                return ownerclass._member_map_[self.name]
            except KeyError:
                raise AttributeError('%r has no attribute %r' % (ownerclass, self.name))
        elif self.fget is None:
            try:
                return ownerclass._member_map_[self.name]
            except KeyError:
                raise AttributeError('%r has no attribute %r' % (ownerclass, self.name)) from None
        else:
            return self.fget(instance)

    def __set__(self, instance, value):
        if self.fset is None:
            raise AttributeError('<enum %r> cannot set attribute %r' % (self.clsname, self.name))
        else:
            return self.fset(instance, value)

    def __delete__(self, instance):
        if self.fdel is None:
            raise AttributeError('<enum %r> cannot delete attribute %r' % (self.clsname, self.name))
        else:
            return self.fdel(instance)

    def __set_name__(self, ownerclass, name):
        self.name = name
        self.clsname = ownerclass.__name__