import logging
import functools
import inspect
import itertools
import sys
import textwrap
import types
from pyomo.common.errors import DeveloperError
class RenamedClass(type):
    """Metaclass to provide a deprecation path for renamed classes

    This metaclass provides a mechanism for renaming old classes while
    still preserving isinstance / issubclass relationships.

    Examples
    --------
    >>> from pyomo.common.deprecation import RenamedClass
    >>> class NewClass(object):
    ...     pass
    >>> class OldClass(metaclass=RenamedClass):
    ...     __renamed__new_class__ = NewClass
    ...     __renamed__version__ = '6.0'

    Deriving from the old class generates a warning:

    >>> class DerivedOldClass(OldClass):
    ...     pass
    WARNING: DEPRECATED: Declaring class 'DerivedOldClass' derived from
        'OldClass'. The class 'OldClass' has been renamed to 'NewClass'.
        (deprecated in 6.0) ...

    As does instantiating the old class:

    >>> old = OldClass()
    WARNING: DEPRECATED: Instantiating class 'OldClass'.  The class
        'OldClass' has been renamed to 'NewClass'.  (deprecated in 6.0) ...

    Finally, `isinstance` and `issubclass` still work, for example:

    >>> isinstance(old, NewClass)
    True
    >>> class NewSubclass(NewClass):
    ...     pass
    >>> new = NewSubclass()
    >>> isinstance(new, OldClass)
    WARNING: DEPRECATED: Checking type relative to 'OldClass'.  The class
        'OldClass' has been renamed to 'NewClass'.  (deprecated in 6.0) ...
    True

    """

    def __new__(cls, name, bases, classdict, *args, **kwargs):
        new_class = classdict.get('__renamed__new_class__', None)
        if new_class is not None:

            def __renamed__new__(cls, *args, **kwargs):
                cls.__renamed__warning__("Instantiating class '%s'." % (cls.__name__,))
                return new_class(*args, **kwargs)
            classdict['__new__'] = __renamed__new__

            def __renamed__warning__(msg):
                version = classdict.get('__renamed__version__')
                remove_in = classdict.get('__renamed__remove_in__')
                deprecation_warning("%s  The class '%s' has been renamed to '%s'." % (msg, name, new_class.__name__), version=version, remove_in=remove_in, calling_frame=_find_calling_frame(1))
            classdict['__renamed__warning__'] = __renamed__warning__
            if not classdict.get('__renamed__version__'):
                raise DeveloperError("Declaring class '%s' using the RenamedClass metaclass, but without specifying the __renamed__version__ class attribute" % (name,))
        renamed_bases = []
        for base in bases:
            new_class = getattr(base, '__renamed__new_class__', None)
            if new_class is not None:
                base.__renamed__warning__("Declaring class '%s' derived from '%s'." % (name, base.__name__))
                base = new_class
                classdict.setdefault('__renamed__new_class__', None)
            if base not in renamed_bases:
                renamed_bases.append(base)
        if new_class is not None and new_class not in renamed_bases:
            renamed_bases.append(new_class)
        if new_class is None and '__renamed__new_class__' not in classdict:
            if not any((hasattr(base, '__renamed__new_class__') for mro in itertools.chain.from_iterable((base.__mro__ for base in renamed_bases)))):
                raise TypeError("Declaring class '%s' using the RenamedClass metaclass, but without specifying the __renamed__new_class__ class attribute" % (name,))
        return super().__new__(cls, name, tuple(renamed_bases), classdict, *args, **kwargs)

    def __instancecheck__(cls, instance):
        return any((cls.__subclasscheck__(c) for c in {type(instance), instance.__class__}))

    def __subclasscheck__(cls, subclass):
        if hasattr(cls, '__renamed__warning__'):
            cls.__renamed__warning__("Checking type relative to '%s'." % (cls.__name__,))
        if subclass is cls:
            return True
        elif getattr(cls, '__renamed__new_class__') is not None:
            return issubclass(subclass, getattr(cls, '__renamed__new_class__'))
        else:
            return super().__subclasscheck__(subclass)