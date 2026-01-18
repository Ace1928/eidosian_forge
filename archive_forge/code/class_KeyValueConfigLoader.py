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
class KeyValueConfigLoader(KVArgParseConfigLoader):
    """Deprecated in traitlets 5.0

    Use KVArgParseConfigLoader
    """

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        warnings.warn('KeyValueConfigLoader is deprecated since Traitlets 5.0. Use KVArgParseConfigLoader instead.', DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)