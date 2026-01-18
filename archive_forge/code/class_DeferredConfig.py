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
class DeferredConfig:
    """Class for deferred-evaluation of config from CLI"""

    def get_value(self, trait: TraitType[t.Any, t.Any]) -> t.Any:
        raise NotImplementedError('Implement in subclasses')

    def _super_repr(self) -> str:
        return super(self.__class__, self).__repr__()