import inspect
from typing import Any, Dict, List, Optional, Union
import torch.nn
from . import utils, variables
from .bytecode_transformation import (
from .codegen import PyCodegen
from .exc import unimplemented
from .source import LocalSource, Source
from .utils import nn_module_new, object_new
from .variables.base import (
def _track_obj(self, source: Source, item: Any, variable: VariableTracker, mutable_cls=MutableSideEffects):
    """Start tracking a new variable for mutation"""
    variable = variable.clone(mutable_local=mutable_cls(source), source=source)
    self.id_to_variable[id(item)] = variable
    self.keepalive.append(item)
    return variable