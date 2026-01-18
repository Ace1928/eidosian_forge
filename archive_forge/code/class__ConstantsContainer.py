from __future__ import division, absolute_import
from functools import partial
from itertools import count
from operator import and_, or_, xor
class _ConstantsContainer(_ConstantsContainerType('', (object,), {})):
    """
    L{_ConstantsContainer} is a class with attributes used as symbolic
    constants.  It is up to subclasses to specify what kind of constants are
    allowed.

    @cvar _constantType: Specified by a L{_ConstantsContainer} subclass to
        specify the type of constants allowed by that subclass.

    @cvar _enumerants: A C{dict} mapping the names of constants (eg
        L{NamedConstant} instances) found in the class definition to those
        instances.
    """
    _constantType = None

    def __new__(cls):
        """
        Classes representing constants containers are not intended to be
        instantiated.

        The class object itself is used directly.
        """
        raise TypeError('%s may not be instantiated.' % (cls.__name__,))

    @classmethod
    def _constantFactory(cls, name, descriptor):
        """
        Construct the value for a new constant to add to this container.

        @param name: The name of the constant to create.

        @param descriptor: An instance of a L{_Constant} subclass (eg
            L{NamedConstant}) which is assigned to C{name}.

        @return: L{NamedConstant} instances have no value apart from identity,
            so return a meaningless dummy value.
        """
        return _unspecified

    @classmethod
    def lookupByName(cls, name):
        """
        Retrieve a constant by its name or raise a C{ValueError} if there is no
        constant associated with that name.

        @param name: A C{str} giving the name of one of the constants defined
            by C{cls}.

        @raise ValueError: If C{name} is not the name of one of the constants
            defined by C{cls}.

        @return: The L{NamedConstant} associated with C{name}.
        """
        if name in cls._enumerants:
            return getattr(cls, name)
        raise ValueError(name)

    @classmethod
    def iterconstants(cls):
        """
        Iteration over a L{Names} subclass results in all of the constants it
        contains.

        @return: an iterator the elements of which are the L{NamedConstant}
            instances defined in the body of this L{Names} subclass.
        """
        constants = cls._enumerants.values()
        return iter(sorted(constants, key=lambda descriptor: descriptor._index))