import inspect
import math
import re
import warnings
from collections import OrderedDict
from copy import deepcopy
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torchvision
from torch import fx, nn
from torch.fx.graph_module import _copy_attr
def _get_leaf_modules_for_ops() -> List[type]:
    members = inspect.getmembers(torchvision.ops)
    result = []
    for _, obj in members:
        if inspect.isclass(obj) and issubclass(obj, torch.nn.Module):
            result.append(obj)
    return result