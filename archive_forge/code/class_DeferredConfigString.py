from __future__ import annotations
import argparse
import copy
import functools
import json
import os
import re
import sys
import typing as t
from logging import Logger
from traitlets.traitlets import Any, Container, Dict, HasTraits, List, TraitType, Undefined
from ..utils import cast_unicode, filefind, warnings
class DeferredConfigString(str, DeferredConfig):
    """Config value for loading config from a string

    Interpretation is deferred until it is loaded into the trait.

    Subclass of str for backward compatibility.

    This class is only used for values that are not listed
    in the configurable classes.

    When config is loaded, `trait.from_string` will be used.

    If an error is raised in `.from_string`,
    the original string is returned.

    .. versionadded:: 5.0
    """

    def get_value(self, trait: TraitType[t.Any, t.Any]) -> t.Any:
        """Get the value stored in this string"""
        s = str(self)
        try:
            return trait.from_string(s)
        except Exception:
            return s

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._super_repr()})'