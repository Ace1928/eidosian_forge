from __future__ import annotations
import argparse
import contextlib
import copy
import enum
import functools
import logging
from typing import Generator
from typing import Sequence
from flake8 import defaults
from flake8 import statistics
from flake8 import utils
from flake8.formatting import base as base_formatter
from flake8.violation import Violation
def _style_guide_for(self, filename: str) -> StyleGuide:
    """Find the StyleGuide for the filename in particular."""
    return max((g for g in self.style_guides if g.applies_to(filename)), key=lambda g: len(g.filename or ''))