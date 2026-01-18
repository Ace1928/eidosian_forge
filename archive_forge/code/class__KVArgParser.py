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
class _KVArgParser(argparse.ArgumentParser):
    """subclass of ArgumentParser where any --Class.trait option is implicitly defined"""

    def parse_known_args(self, args: t.Sequence[str] | None=None, namespace: argparse.Namespace | None=None) -> tuple[argparse.Namespace | None, list[str]]:
        for container in (self, self._optionals):
            container._option_string_actions = _DefaultOptionDict(container._option_string_actions)
        return super().parse_known_args(args, namespace)