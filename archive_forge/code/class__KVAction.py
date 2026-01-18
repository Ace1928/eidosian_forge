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
class _KVAction(argparse.Action):
    """Custom argparse action for handling --Class.trait=x

    Always
    """

    def __call__(self, parser: argparse.ArgumentParser, namespace: dict[str, t.Any], values: t.Sequence[t.Any], option_string: str | None=None) -> None:
        if isinstance(values, str):
            values = [values]
        values = ['-' if v is _DASH_REPLACEMENT else v for v in values]
        items = getattr(namespace, self.dest, None)
        if items is None:
            items = DeferredConfigList()
        else:
            items = DeferredConfigList(items)
        items.extend(values)
        setattr(namespace, self.dest, items)