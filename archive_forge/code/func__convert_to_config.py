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
def _convert_to_config(self) -> None:
    """self.parsed_data->self.config, parse unrecognized extra args via KVLoader."""
    extra_args = self.extra_args
    for lhs, rhs in vars(self.parsed_data).items():
        if lhs == 'extra_args':
            self.extra_args = ['-' if a == _DASH_REPLACEMENT else a for a in rhs] + extra_args
            continue
        if lhs == '_flags':
            continue
        lhs = lhs.replace(_DOT_REPLACEMENT, '.')
        if '.' not in lhs:
            self._handle_unrecognized_alias(lhs)
            trait = None
        if isinstance(rhs, list):
            rhs = DeferredConfigList(rhs)
        elif isinstance(rhs, str):
            rhs = DeferredConfigString(rhs)
        trait = self.argparse_traits.get(lhs)
        if trait:
            trait = trait[0]
        try:
            self._exec_config_str(lhs, rhs, trait)
        except Exception as e:
            if isinstance(rhs, DeferredConfig):
                rhs = rhs._super_repr()
            raise ArgumentError(f'Error loading argument {lhs}={rhs}, {e}') from e
    for subc in self.parsed_data._flags:
        self._load_flag(subc)