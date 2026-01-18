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
def _load_flag(self, cfg: t.Any) -> None:
    """update self.config from a flag, which can be a dict or Config"""
    if isinstance(cfg, (dict, Config)):
        for sec, c in cfg.items():
            self.config[sec].update(c)
    else:
        raise TypeError('Invalid flag: %r' % cfg)