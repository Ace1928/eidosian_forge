import functools
import warnings
from typing import (
import torch
import optree
from optree import PyTreeSpec  # direct import for type annotations
class _DummyLeaf:

    def __repr__(self) -> str:
        return '*'