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
def _set_default_tracer_kwargs(original_tr_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    default_autowrap_modules = (math, torchvision.ops)
    default_leaf_modules = _get_leaf_modules_for_ops()
    result_tracer_kwargs = {} if original_tr_kwargs is None else original_tr_kwargs
    result_tracer_kwargs['autowrap_modules'] = tuple(set(result_tracer_kwargs['autowrap_modules'] + default_autowrap_modules)) if 'autowrap_modules' in result_tracer_kwargs else default_autowrap_modules
    result_tracer_kwargs['leaf_modules'] = list(set(result_tracer_kwargs['leaf_modules'] + default_leaf_modules)) if 'leaf_modules' in result_tracer_kwargs else default_leaf_modules
    return result_tracer_kwargs