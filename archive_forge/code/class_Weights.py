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
@dataclass
class Weights:
    """
    This class is used to group important attributes associated with the pre-trained weights.

    Args:
        url (str): The location where we find the weights.
        transforms (Callable): A callable that constructs the preprocessing method (or validation preset transforms)
            needed to use the model. The reason we attach a constructor method rather than an already constructed
            object is because the specific object might have memory and thus we want to delay initialization until
            needed.
        meta (Dict[str, Any]): Stores meta-data related to the weights of the model and its configuration. These can be
            informative attributes (for example the number of parameters/flops, recipe link/methods used in training
            etc), configuration parameters (for example the `num_classes`) needed to construct the model or important
            meta-data (for example the `classes` of a classification model) needed to use the model.
    """
    url: str
    transforms: Callable
    meta: Dict[str, Any]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Weights):
            return NotImplemented
        if self.url != other.url:
            return False
        if self.meta != other.meta:
            return False
        if isinstance(self.transforms, partial) and isinstance(other.transforms, partial):
            return self.transforms.func == other.transforms.func and self.transforms.args == other.transforms.args and (self.transforms.keywords == other.transforms.keywords)
        else:
            return self.transforms == other.transforms