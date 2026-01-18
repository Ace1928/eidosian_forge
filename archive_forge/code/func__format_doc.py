from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, Optional, Tuple, TypeVar, Union
import torch
import torio
from torch.utils._pytree import tree_map
def _format_doc(**kwargs):

    def decorator(obj):
        obj.__doc__ = obj.__doc__.format(**kwargs)
        return obj
    return decorator