import fnmatch
import importlib
import inspect
import sys
from dataclasses import dataclass
from enum import Enum
from functools import partial
from inspect import signature
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set, Type, TypeVar, Union
from torch import nn
from .._internally_replaced_utils import load_state_dict_from_url
def _get_enum_from_fn(fn: Callable) -> Type[WeightsEnum]:
    """
    Internal method that gets the weight enum of a specific model builder method.

    Args:
        fn (Callable): The builder method used to create the model.
    Returns:
        WeightsEnum: The requested weight enum.
    """
    sig = signature(fn)
    if 'weights' not in sig.parameters:
        raise ValueError("The method is missing the 'weights' argument.")
    ann = signature(fn).parameters['weights'].annotation
    weights_enum = None
    if isinstance(ann, type) and issubclass(ann, WeightsEnum):
        weights_enum = ann
    else:
        for t in ann.__args__:
            if isinstance(t, type) and issubclass(t, WeightsEnum):
                weights_enum = t
                break
    if weights_enum is None:
        raise ValueError("The WeightsEnum class for the specific method couldn't be retrieved. Make sure the typing info is correct.")
    return weights_enum