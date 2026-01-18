from __future__ import annotations
import logging
import typing as t
from copy import deepcopy
from textwrap import dedent
from traitlets.traitlets import (
from traitlets.utils import warnings
from traitlets.utils.bunch import Bunch
from traitlets.utils.text import indent, wrap_paragraphs
from .loader import Config, DeferredConfig, LazyConfigValue, _is_section_key
class SingletonConfigurable(LoggingConfigurable):
    """A configurable that only allows one instance.

    This class is for classes that should only have one instance of itself
    or *any* subclass. To create and retrieve such a class use the
    :meth:`SingletonConfigurable.instance` method.
    """
    _instance = None

    @classmethod
    def _walk_mro(cls) -> t.Generator[type[SingletonConfigurable], None, None]:
        """Walk the cls.mro() for parent classes that are also singletons

        For use in instance()
        """
        for subclass in cls.mro():
            if issubclass(cls, subclass) and issubclass(subclass, SingletonConfigurable) and (subclass != SingletonConfigurable):
                yield subclass

    @classmethod
    def clear_instance(cls) -> None:
        """unset _instance for this class and singleton parents."""
        if not cls.initialized():
            return
        for subclass in cls._walk_mro():
            if isinstance(subclass._instance, cls):
                subclass._instance = None

    @classmethod
    def instance(cls: type[CT], *args: t.Any, **kwargs: t.Any) -> CT:
        """Returns a global instance of this class.

        This method create a new instance if none have previously been created
        and returns a previously created instance is one already exists.

        The arguments and keyword arguments passed to this method are passed
        on to the :meth:`__init__` method of the class upon instantiation.

        Examples
        --------
        Create a singleton class using instance, and retrieve it::

            >>> from traitlets.config.configurable import SingletonConfigurable
            >>> class Foo(SingletonConfigurable): pass
            >>> foo = Foo.instance()
            >>> foo == Foo.instance()
            True

        Create a subclass that is retrieved using the base class instance::

            >>> class Bar(SingletonConfigurable): pass
            >>> class Bam(Bar): pass
            >>> bam = Bam.instance()
            >>> bam == Bar.instance()
            True
        """
        if cls._instance is None:
            inst = cls(*args, **kwargs)
            for subclass in cls._walk_mro():
                subclass._instance = inst
        if isinstance(cls._instance, cls):
            return cls._instance
        else:
            raise MultipleInstanceError(f"An incompatible sibling of '{cls.__name__}' is already instantiated as singleton: {type(cls._instance).__name__}")

    @classmethod
    def initialized(cls) -> bool:
        """Has an instance been created?"""
        return hasattr(cls, '_instance') and cls._instance is not None