import uuid
from collections import OrderedDict
from functools import wraps
from typing import Callable, Dict, List, Optional, Type
import torch.nn as nn
from torch.distributed._composable_state import _State

    Get an ``OrderedDict`` of composable APIs that have been applied to the
    ``module``, indexed by the API name. If no API has been applied, then this
    returns ``None``.
    