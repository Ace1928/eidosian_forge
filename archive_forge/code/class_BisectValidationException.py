import functools
import logging
import math
import operator
import sympy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import torch
import torch.fx
import torch.fx.traceback as fx_traceback
from torch._dynamo.exc import TorchDynamoException
from torch.fx.node import Argument, Target
from torch.utils._sympy.interp import sympy_interp
from torch.fx.experimental import _config as config
class BisectValidationException(TorchDynamoException):

    def __init__(self, validation_exc, expr, failed_action, traced_node):
        self.msg = f'translation validation failed when {failed_action}: {expr}'
        self.details = f'Failure occurred while running node:\n    {traced_node.format_node()}\n\n{validation_exc.details}'

    def __str__(self):
        return f'{self.msg}\n\n{self.details}'