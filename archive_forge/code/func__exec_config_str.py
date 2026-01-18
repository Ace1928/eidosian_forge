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
def _exec_config_str(self, lhs: t.Any, rhs: t.Any, trait: TraitType[t.Any, t.Any] | None=None) -> None:
    """execute self.config.<lhs> = <rhs>

        * expands ~ with expanduser
        * interprets value with trait if available
        """
    value = rhs
    if isinstance(value, DeferredConfig):
        if trait:
            value = value.get_value(trait)
        elif isinstance(rhs, DeferredConfigList) and len(rhs) == 1:
            value = DeferredConfigString(os.path.expanduser(rhs[0]))
    elif trait:
        value = trait.from_string(value)
    else:
        value = DeferredConfigString(value)
    *path, key = lhs.split('.')
    section = self.config
    for part in path:
        section = section[part]
    section[key] = value
    return