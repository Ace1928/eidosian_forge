from abc import abstractmethod, ABCMeta
import collections
from collections import defaultdict
import collections.abc
import contextlib
import functools
import operator
import re as stdlib_re  # Avoid confusion with the re we export.
import sys
import types
import warnings
from types import WrapperDescriptorType, MethodWrapperType, MethodDescriptorType, GenericAlias
class _BoundVarianceMixin:
    """Mixin giving __init__ bound and variance arguments.

    This is used by TypeVar and ParamSpec, which both employ the notions of
    a type 'bound' (restricting type arguments to be a subtype of some
    specified type) and type 'variance' (determining subtype relations between
    generic types).
    """

    def __init__(self, bound, covariant, contravariant):
        """Used to setup TypeVars and ParamSpec's bound, covariant and
        contravariant attributes.
        """
        if covariant and contravariant:
            raise ValueError('Bivariant types are not supported.')
        self.__covariant__ = bool(covariant)
        self.__contravariant__ = bool(contravariant)
        if bound:
            self.__bound__ = _type_check(bound, 'Bound must be a type.')
        else:
            self.__bound__ = None

    def __or__(self, right):
        return Union[self, right]

    def __ror__(self, left):
        return Union[left, self]

    def __repr__(self):
        if self.__covariant__:
            prefix = '+'
        elif self.__contravariant__:
            prefix = '-'
        else:
            prefix = '~'
        return prefix + self.__name__