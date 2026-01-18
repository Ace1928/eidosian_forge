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
def _add_arguments(self, aliases: t.Any, flags: t.Any, classes: t.Any) -> None:
    alias_flags: dict[str, t.Any] = {}
    argparse_kwds: dict[str, t.Any]
    argparse_traits: dict[str, t.Any]
    paa = self.parser.add_argument
    self.parser.set_defaults(_flags=[])
    paa('extra_args', nargs='*')
    self.argparse_traits = argparse_traits = {}
    for cls in classes:
        for traitname, trait in cls.class_traits(config=True).items():
            argname = f'{cls.__name__}.{traitname}'
            argparse_kwds = {'type': str}
            if isinstance(trait, (Container, Dict)):
                multiplicity = trait.metadata.get('multiplicity', 'append')
                if multiplicity == 'append':
                    argparse_kwds['action'] = multiplicity
                else:
                    argparse_kwds['nargs'] = multiplicity
            argparse_traits[argname] = (trait, argparse_kwds)
    for keys, (value, fhelp) in flags.items():
        if not isinstance(keys, tuple):
            keys = (keys,)
        for key in keys:
            if key in aliases:
                alias_flags[aliases[key]] = value
                continue
            keys = ('-' + key, '--' + key) if len(key) == 1 else ('--' + key,)
            paa(*keys, action=_FlagAction, flag=value, help=fhelp)
    for keys, traitname in aliases.items():
        if not isinstance(keys, tuple):
            keys = (keys,)
        for key in keys:
            argparse_kwds = {'type': str, 'dest': traitname.replace('.', _DOT_REPLACEMENT), 'metavar': traitname}
            argcompleter = None
            if traitname in argparse_traits:
                trait, kwds = argparse_traits[traitname]
                argparse_kwds.update(kwds)
                if 'action' in argparse_kwds and traitname in alias_flags:
                    raise ArgumentError(f"The alias `{key}` for the 'append' sequence config-trait `{traitname}` cannot be also a flag!'")
                argcompleter = trait.metadata.get('argcompleter') or getattr(trait, 'argcompleter', None)
            if traitname in alias_flags:
                argparse_kwds.setdefault('nargs', '?')
                argparse_kwds['action'] = _FlagAction
                argparse_kwds['flag'] = alias_flags[traitname]
                argparse_kwds['alias'] = traitname
            keys = ('-' + key, '--' + key) if len(key) == 1 else ('--' + key,)
            action = paa(*keys, **argparse_kwds)
            if argcompleter is not None:
                action.completer = functools.partial(argcompleter, key=key)