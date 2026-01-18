import operator
from typing import Dict, List
import torch
from torch._dynamo.source import GetItemSource
from .. import variables
from ..exc import unimplemented, UserError, UserErrorType
from ..guards import GuardBuilder, install_guard
from ..utils import np
from .base import typestr, VariableTracker

        Need this when adding a BaseListVariable and a ConstantVariable together.
        Happens in detectron2.
        