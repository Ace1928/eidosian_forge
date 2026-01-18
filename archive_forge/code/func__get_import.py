from __future__ import annotations
import ast
import builtins
import sys
import typing
from ast import (
from collections import defaultdict
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast, overload
def _get_import(self, module: str, name: str) -> Name:
    memo = self._memo if self._target_path else self._module_memo
    return memo.get_import(module, name)