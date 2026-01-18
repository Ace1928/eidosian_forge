from __future__ import annotations
import argparse
import dataclasses
import enum
import functools
import itertools
import json
import shlex
from typing import (
import rich.markup
import shtab
from . import _fields, _instantiators, _resolver, _strings
from ._typing import TypeForm
from .conf import _markers
If the instantiator is set to `None`, even after all argument
        transformations, it means that we don't have a valid instantiator for an
        argument. We then mark the argument as 'fixed', with a value always equal to the
        field default.