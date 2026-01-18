from __future__ import (absolute_import, division, print_function)
from abc import ABCMeta
from collections.abc import Container, Mapping, Sequence, Set
from ansible.module_utils.common.collections import ImmutableDict
from ansible.module_utils.six import add_metaclass, binary_type, text_type
from ansible.utils.singleton import Singleton
class _ABCSingleton(Singleton, ABCMeta):
    """
    Combine ABCMeta based classes with Singleton based classes

    Combine Singleton and ABCMeta so we have a metaclass that unambiguously knows which can override
    the other.  Useful for making new types of containers which are also Singletons.
    """
    pass