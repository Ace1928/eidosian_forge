from __future__ import annotations
from pathlib import Path
import argparse
import enum
import sys
import stat
import time
import abc
import platform, subprocess, operator, os, shlex, shutil, re
import collections
from functools import lru_cache, wraps, total_ordering
from itertools import tee
from tempfile import TemporaryDirectory, NamedTemporaryFile
import typing as T
import textwrap
import pickle
import errno
import json
from mesonbuild import mlog
from .core import MesonException, HoldableObject
from glob import glob
@total_ordering
class OptionKey:
    """Represents an option key in the various option dictionaries.

    This provides a flexible, powerful way to map option names from their
    external form (things like subproject:build.option) to something that
    internally easier to reason about and produce.
    """
    __slots__ = ['name', 'subproject', 'machine', 'lang', '_hash', 'type', 'module']
    name: str
    subproject: str
    machine: MachineChoice
    lang: T.Optional[str]
    _hash: int
    type: OptionType
    module: T.Optional[str]

    def __init__(self, name: str, subproject: str='', machine: MachineChoice=MachineChoice.HOST, lang: T.Optional[str]=None, module: T.Optional[str]=None, _type: T.Optional[OptionType]=None):
        object.__setattr__(self, 'name', name)
        object.__setattr__(self, 'subproject', subproject)
        object.__setattr__(self, 'machine', machine)
        object.__setattr__(self, 'lang', lang)
        object.__setattr__(self, 'module', module)
        object.__setattr__(self, '_hash', hash((name, subproject, machine, lang, module)))
        if _type is None:
            _type = _classify_argument(self)
        object.__setattr__(self, 'type', _type)

    def __setattr__(self, key: str, value: T.Any) -> None:
        raise AttributeError('OptionKey instances do not support mutation.')

    def __getstate__(self) -> T.Dict[str, T.Any]:
        return {'name': self.name, 'subproject': self.subproject, 'machine': self.machine, 'lang': self.lang, '_type': self.type, 'module': self.module}

    def __setstate__(self, state: T.Dict[str, T.Any]) -> None:
        """De-serialize the state of a pickle.

        This is very clever. __init__ is not a constructor, it's an
        initializer, therefore it's safe to call more than once. We create a
        state in the custom __getstate__ method, which is valid to pass
        splatted to the initializer.
        """
        self.__init__(**state)

    def __hash__(self) -> int:
        return self._hash

    def _to_tuple(self) -> T.Tuple[str, OptionType, str, str, MachineChoice, str]:
        return (self.subproject, self.type, self.lang or '', self.module or '', self.machine, self.name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, OptionKey):
            return self._to_tuple() == other._to_tuple()
        return NotImplemented

    def __lt__(self, other: object) -> bool:
        if isinstance(other, OptionKey):
            return self._to_tuple() < other._to_tuple()
        return NotImplemented

    def __str__(self) -> str:
        out = self.name
        if self.lang:
            out = f'{self.lang}_{out}'
        if self.machine is MachineChoice.BUILD:
            out = f'build.{out}'
        if self.module:
            out = f'{self.module}.{out}'
        if self.subproject:
            out = f'{self.subproject}:{out}'
        return out

    def __repr__(self) -> str:
        return f'OptionKey({self.name!r}, {self.subproject!r}, {self.machine!r}, {self.lang!r}, {self.module!r}, {self.type!r})'

    @classmethod
    def from_string(cls, raw: str) -> 'OptionKey':
        """Parse the raw command line format into a three part tuple.

        This takes strings like `mysubproject:build.myoption` and Creates an
        OptionKey out of them.
        """
        try:
            subproject, raw2 = raw.split(':')
        except ValueError:
            subproject, raw2 = ('', raw)
        module = None
        for_machine = MachineChoice.HOST
        try:
            prefix, raw3 = raw2.split('.')
            if prefix == 'build':
                for_machine = MachineChoice.BUILD
            else:
                module = prefix
        except ValueError:
            raw3 = raw2
        from ..compilers import all_languages
        if any((raw3.startswith(f'{l}_') for l in all_languages)):
            lang, opt = raw3.split('_', 1)
        else:
            lang, opt = (None, raw3)
        assert ':' not in opt
        assert '.' not in opt
        return cls(opt, subproject, for_machine, lang, module)

    def evolve(self, name: T.Optional[str]=None, subproject: T.Optional[str]=None, machine: T.Optional[MachineChoice]=None, lang: T.Optional[str]='', module: T.Optional[str]='') -> 'OptionKey':
        """Create a new copy of this key, but with altered members.

        For example:
        >>> a = OptionKey('foo', '', MachineChoice.Host)
        >>> b = OptionKey('foo', 'bar', MachineChoice.Host)
        >>> b == a.evolve(subproject='bar')
        True
        """
        return OptionKey(name if name is not None else self.name, subproject if subproject is not None else self.subproject, machine if machine is not None else self.machine, lang if lang != '' else self.lang, module if module != '' else self.module)

    def as_root(self) -> 'OptionKey':
        """Convenience method for key.evolve(subproject='')."""
        return self.evolve(subproject='')

    def as_build(self) -> 'OptionKey':
        """Convenience method for key.evolve(machine=MachineChoice.BUILD)."""
        return self.evolve(machine=MachineChoice.BUILD)

    def as_host(self) -> 'OptionKey':
        """Convenience method for key.evolve(machine=MachineChoice.HOST)."""
        return self.evolve(machine=MachineChoice.HOST)

    def is_backend(self) -> bool:
        """Convenience method to check if this is a backend option."""
        return self.type is OptionType.BACKEND

    def is_builtin(self) -> bool:
        """Convenience method to check if this is a builtin option."""
        return self.type is OptionType.BUILTIN

    def is_compiler(self) -> bool:
        """Convenience method to check if this is a builtin option."""
        return self.type is OptionType.COMPILER

    def is_project(self) -> bool:
        """Convenience method to check if this is a project option."""
        return self.type is OptionType.PROJECT

    def is_base(self) -> bool:
        """Convenience method to check if this is a base option."""
        return self.type is OptionType.BASE