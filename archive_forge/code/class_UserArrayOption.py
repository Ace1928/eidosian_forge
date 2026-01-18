from __future__ import annotations
import copy
from . import mlog, mparser
import pickle, os, uuid
import sys
from itertools import chain
from pathlib import PurePath
from collections import OrderedDict, abc
from dataclasses import dataclass
from .mesonlib import (
from .wrap import WrapMode
import ast
import argparse
import configparser
import enum
import shlex
import typing as T
class UserArrayOption(UserOption[T.List[str]]):

    def __init__(self, description: str, value: T.Union[str, T.List[str]], split_args: bool=False, allow_dups: bool=False, yielding: bool=DEFAULT_YIELDING, choices: T.Optional[T.List[str]]=None, deprecated: T.Union[bool, str, T.Dict[str, str], T.List[str]]=False):
        super().__init__(description, choices if choices is not None else [], yielding, deprecated)
        self.split_args = split_args
        self.allow_dups = allow_dups
        self.set_value(value)

    @staticmethod
    def listify_value(value: T.Union[str, T.List[str]], shlex_split_args: bool=False) -> T.List[str]:
        if isinstance(value, str):
            if value.startswith('['):
                try:
                    newvalue = ast.literal_eval(value)
                except ValueError:
                    raise MesonException(f'malformed option {value}')
            elif value == '':
                newvalue = []
            elif shlex_split_args:
                newvalue = split_args(value)
            else:
                newvalue = [v.strip() for v in value.split(',')]
        elif isinstance(value, list):
            newvalue = value
        else:
            raise MesonException(f'"{value}" should be a string array, but it is not')
        return newvalue

    def listify(self, value: T.Any) -> T.List[T.Any]:
        return self.listify_value(value, self.split_args)

    def validate_value(self, value: T.Union[str, T.List[str]]) -> T.List[str]:
        newvalue = self.listify(value)
        if not self.allow_dups and len(set(newvalue)) != len(newvalue):
            msg = 'Duplicated values in array option is deprecated. This will become a hard error in the future.'
            mlog.deprecation(msg)
        for i in newvalue:
            if not isinstance(i, str):
                raise MesonException(f'String array element "{newvalue!s}" is not a string.')
        if self.choices:
            bad = [x for x in newvalue if x not in self.choices]
            if bad:
                raise MesonException('Options "{}" are not in allowed choices: "{}"'.format(', '.join(bad), ', '.join(self.choices)))
        return newvalue

    def extend_value(self, value: T.Union[str, T.List[str]]) -> None:
        """Extend the value with an additional value."""
        new = self.validate_value(value)
        self.set_value(self.value + new)