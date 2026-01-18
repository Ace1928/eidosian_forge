from __future__ import annotations
import contextlib
import enum
import inspect
import os
import re
import sys
import types
import typing as t
from ast import literal_eval
from .utils.bunch import Bunch
from .utils.descriptions import add_article, class_of, describe, repr_type
from .utils.getargspec import getargspec
from .utils.importstring import import_item
from .utils.sentinel import Sentinel
from .utils.warnings import deprecated_method, should_warn, warn
from all trait attributes.
class MetaHasDescriptors(type):
    """A metaclass for HasDescriptors.

    This metaclass makes sure that any TraitType class attributes are
    instantiated and sets their name attribute.
    """

    def __new__(mcls: type[MetaHasDescriptors], name: str, bases: tuple[type, ...], classdict: dict[str, t.Any], **kwds: t.Any) -> MetaHasDescriptors:
        """Create the HasDescriptors class."""
        for k, v in classdict.items():
            if inspect.isclass(v) and issubclass(v, TraitType):
                warn('Traits should be given as instances, not types (for example, `Int()`, not `Int`). Passing types is deprecated in traitlets 4.1.', DeprecationWarning, stacklevel=2)
                classdict[k] = v()
        return super().__new__(mcls, name, bases, classdict, **kwds)

    def __init__(cls, name: str, bases: tuple[type, ...], classdict: dict[str, t.Any], **kwds: t.Any) -> None:
        """Finish initializing the HasDescriptors class."""
        super().__init__(name, bases, classdict, **kwds)
        cls.setup_class(classdict)

    def setup_class(cls: MetaHasDescriptors, classdict: dict[str, t.Any]) -> None:
        """Setup descriptor instance on the class

        This sets the :attr:`this_class` and :attr:`name` attributes of each
        BaseDescriptor in the class dict of the newly created ``cls`` before
        calling their :attr:`class_init` method.
        """
        cls._descriptors = []
        cls._instance_inits: list[t.Any] = []
        for k, v in classdict.items():
            if isinstance(v, BaseDescriptor):
                v.class_init(cls, k)
        for _, v in getmembers(cls):
            if isinstance(v, BaseDescriptor):
                v.subclass_init(cls)
                cls._descriptors.append(v)