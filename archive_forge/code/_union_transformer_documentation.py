from __future__ import annotations
from ast import (
from ast import Tuple as ASTTuple
from types import CodeType
from typing import Any, Dict, FrozenSet, List, Set, Tuple, Union

Transforms lazily evaluated PEP 604 unions into typing.Unions, for compatibility with
Python versions older than 3.10.
