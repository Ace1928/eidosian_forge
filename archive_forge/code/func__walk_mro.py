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
@classmethod
def _walk_mro(cls) -> t.Generator[type[SingletonConfigurable], None, None]:
    """Walk the cls.mro() for parent classes that are also singletons

        For use in instance()
        """
    for subclass in cls.mro():
        if issubclass(cls, subclass) and issubclass(subclass, SingletonConfigurable) and (subclass != SingletonConfigurable):
            yield subclass