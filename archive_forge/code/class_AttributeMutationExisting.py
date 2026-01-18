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
class AttributeMutationExisting(AttributeMutation):

    def __init__(self, source: Source):
        super().__init__(MutableLocalSource.Existing, source)
        self.source = source