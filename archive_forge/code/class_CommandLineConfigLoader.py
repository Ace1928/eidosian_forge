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
class CommandLineConfigLoader(ConfigLoader):
    """A config loader for command line arguments.

    As we add more command line based loaders, the common logic should go
    here.
    """

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

    def _load_flag(self, cfg: t.Any) -> None:
        """update self.config from a flag, which can be a dict or Config"""
        if isinstance(cfg, (dict, Config)):
            for sec, c in cfg.items():
                self.config[sec].update(c)
        else:
            raise TypeError('Invalid flag: %r' % cfg)