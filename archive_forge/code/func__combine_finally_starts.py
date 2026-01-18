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
def _combine_finally_starts(self, starts: set[ArcStart], exits: set[ArcStart]) -> set[ArcStart]:
    """Helper for building the cause of `finally` branches.

        "finally" clauses might not execute their exits, and the causes could
        be due to a failure to execute any of the exits in the try block. So
        we use the causes from `starts` as the causes for `exits`.
        """
    causes = []
    for start in sorted(starts):
        if start.cause:
            causes.append(start.cause.format(lineno=start.lineno))
    cause = ' or '.join(causes)
    exits = {ArcStart(xit.lineno, cause) for xit in exits}
    return exits