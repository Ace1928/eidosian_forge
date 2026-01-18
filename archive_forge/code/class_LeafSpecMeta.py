import functools
import warnings
from typing import (
import torch
import optree
from optree import PyTreeSpec  # direct import for type annotations
class LeafSpecMeta(type(TreeSpec)):

    def __instancecheck__(self, instance: object) -> bool:
        return isinstance(instance, TreeSpec) and instance.is_leaf()