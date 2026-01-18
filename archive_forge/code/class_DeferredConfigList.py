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
class DeferredConfigList(t.List[t.Any], DeferredConfig):
    """Config value for loading config from a list of strings

    Interpretation is deferred until it is loaded into the trait.

    This class is only used for values that are not listed
    in the configurable classes.

    When config is loaded, `trait.from_string_list` will be used.

    If an error is raised in `.from_string_list`,
    the original string list is returned.

    .. versionadded:: 5.0
    """

    def get_value(self, trait: TraitType[t.Any, t.Any]) -> t.Any:
        """Get the value stored in this string"""
        if hasattr(trait, 'from_string_list'):
            src = list(self)
            cast = trait.from_string_list
        else:
            if len(self) > 1:
                raise ValueError(f'{trait.name} only accepts one value, got {len(self)}: {list(self)}')
            src = self[0]
            cast = trait.from_string
        try:
            return cast(src)
        except Exception:
            return src

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._super_repr()})'