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

        This blocks names from being collected from a module-level
        "if typing.TYPE_CHECKING:" block, so that they won't be type checked.

        