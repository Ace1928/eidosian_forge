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
def _find_file(self) -> None:
    """Try to find the file by searching the paths."""
    self.full_filename = filefind(self.filename, self.path)