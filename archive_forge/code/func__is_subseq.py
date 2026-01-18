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
def _is_subseq(x, y):
    """Check if y is a subsequence of x
    https://stackoverflow.com/a/24017747/4391249
    """
    iter_x = iter(x)
    return all((any((x_item == y_item for x_item in iter_x)) for y_item in y))