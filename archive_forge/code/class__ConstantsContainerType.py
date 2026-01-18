from __future__ import division, absolute_import
from functools import partial
from itertools import count
from operator import and_, or_, xor
class _ConstantsContainerType(type):
    """
    L{_ConstantsContainerType} is a metaclass for creating constants container
    classes.
    """

    def __new__(self, name, bases, attributes):
        """
        Create a new constants container class.

        If C{attributes} includes a value of C{None} for the C{"_constantType"}
        key, the new class will not be initialized as a constants container and
        it will behave as a normal class.

        @param name: The name of the container class.
        @type name: L{str}

        @param bases: A tuple of the base classes for the new container class.
        @type bases: L{tuple} of L{_ConstantsContainerType} instances

        @param attributes: The attributes of the new container class, including
            any constants it is to contain.
        @type attributes: L{dict}
        """
        cls = super(_ConstantsContainerType, self).__new__(self, name, bases, attributes)
        constantType = getattr(cls, '_constantType', None)
        if constantType is None:
            return cls
        constants = []
        for name, descriptor in attributes.items():
            if isinstance(descriptor, cls._constantType):
                if descriptor._container is not None:
                    raise ValueError('Cannot use %s as the value of an attribute on %s' % (descriptor, cls.__name__))
                constants.append((descriptor._index, name, descriptor))
        enumerants = {}
        for index, enumerant, descriptor in sorted(constants):
            value = cls._constantFactory(enumerant, descriptor)
            descriptor._realize(cls, enumerant, value)
            enumerants[enumerant] = descriptor
        cls._enumerants = enumerants
        return cls