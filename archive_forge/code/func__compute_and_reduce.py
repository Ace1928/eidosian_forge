from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, Hashable, Iterable, Iterator, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import ModuleDict
from typing_extensions import Literal
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import _flatten_dict, allclose
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE, plot_single_or_multi_val
def _compute_and_reduce(self, method_name: Literal['compute', 'forward'], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Compute result from collection and reduce into a single dictionary.

        Args:
            method_name: The method to call on each metric in the collection.
                Should be either `compute` or `forward`.
            args: Positional arguments to pass to each metric (if method_name is `forward`)
            kwargs: Keyword arguments to pass to each metric (if method_name is `forward`)

        Raises:
            ValueError:
                If method_name is not `compute` or `forward`.

        """
    result = {}
    for k, m in self.items(keep_base=True, copy_state=False):
        if method_name == 'compute':
            res = m.compute()
        elif method_name == 'forward':
            res = m(*args, **m._filter_kwargs(**kwargs))
        else:
            raise ValueError("method_name should be either 'compute' or 'forward', but got {method_name}")
        result[k] = res
    _, duplicates = _flatten_dict(result)
    flattened_results = {}
    for k, m in self.items(keep_base=True, copy_state=False):
        res = result[k]
        if isinstance(res, dict):
            for key, v in res.items():
                if duplicates:
                    stripped_k = k.replace(getattr(m, 'prefix', ''), '')
                    stripped_k = stripped_k.replace(getattr(m, 'postfix', ''), '')
                    key = f'{stripped_k}_{key}'
                if getattr(m, '_from_collection', None) and m.prefix is not None:
                    key = f'{m.prefix}{key}'
                if getattr(m, '_from_collection', None) and m.postfix is not None:
                    key = f'{key}{m.postfix}'
                flattened_results[key] = v
        else:
            flattened_results[k] = res
    return {self._set_name(k): v for k, v in flattened_results.items()}