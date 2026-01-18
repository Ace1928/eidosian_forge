from __future__ import annotations
import ast
import collections
import os
import re
import sys
import token
import tokenize
from dataclasses import dataclass
from types import CodeType
from typing import (
from coverage import env
from coverage.bytecode import code_objects
from coverage.debug import short_stack
from coverage.exceptions import NoSource, NotPython
from coverage.misc import join_regex, nice_pair
from coverage.phystokens import generate_tokens
from coverage.types import TArc, TLineNo
@dataclass(frozen=True, order=True)
class ArcStart:
    """The information needed to start an arc.

    `lineno` is the line number the arc starts from.

    `cause` is an English text fragment used as the `startmsg` for
    AstArcAnalyzer.missing_arc_fragments.  It will be used to describe why an
    arc wasn't executed, so should fit well into a sentence of the form,
    "Line 17 didn't run because {cause}."  The fragment can include "{lineno}"
    to have `lineno` interpolated into it.

    """
    lineno: TLineNo
    cause: str = ''