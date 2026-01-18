from __future__ import annotations
from abc import ABCMeta
from collections import defaultdict
from typing import TYPE_CHECKING
from weakref import WeakValueDictionary
from ..exceptions import PlotnineError
class RegistryHierarchyMeta(type):
    """
    Create a class that registers subclasses and the Hierarchy

    The class has gets two properties:

    1. `_registry` a dictionary of all the subclasses of the
       base class. The keys are the names of the classes and
       the values are the class objects.
    2. `_hierarchy` a dictionary (default) that holds the
       inheritance hierarchy of each class. Each key is a class
       and the value is a list of classes. The first name in the
       list is that of the key class.

    The goal of the `_hierarchy` object to facilitate the
    lookup of themeable properties taking into consideration the
    inheritance hierarchy. For example if `strip_text_x` inherits
    from `strip_text` which inherits from `text`, then if a property
    of `strip_text_x` is requested, the lookup should fallback to
    the other two if `strip_text_x` is not present or is missing
    the requested property.
    """

    def __init__(cls, name, bases, namespace):
        if not hasattr(cls, '_registry'):
            cls._registry = {}
            cls._hierarchy = defaultdict(list)
        else:
            cls._registry[name] = cls
            cls._hierarchy[name].append(name)
            for base in bases:
                for base2 in base.mro()[:-2]:
                    cls._hierarchy[base2.__name__].append(name)