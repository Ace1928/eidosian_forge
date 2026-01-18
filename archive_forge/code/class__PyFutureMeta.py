from __future__ import annotations
from typing import cast, Callable, Generic, List, Optional, Type, TypeVar, Union
import torch
class _PyFutureMeta(type(torch._C.Future), type(Generic)):
    pass