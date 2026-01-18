import inspect
import logging
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch.nn.utils.prune as pytorch_prune
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor, nn
from typing_extensions import TypedDict, override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_debug, rank_zero_only
def _create_pruning_fn(self, pruning_fn: str, **kwargs: Any) -> Union[Callable, pytorch_prune.BasePruningMethod]:
    """This function takes `pruning_fn`, a function name.

        IF use_global_unstructured, pruning_fn will be resolved into its associated ``PyTorch BasePruningMethod`` ELSE,
        pruning_fn will be resolved into its function counterpart from `torch.nn.utils.prune`.

        """
    pruning_meth = _PYTORCH_PRUNING_METHOD[pruning_fn] if self._use_global_unstructured else _PYTORCH_PRUNING_FUNCTIONS[pruning_fn]
    assert callable(pruning_meth), 'Selected pruning method is not callable'
    if self._use_global_unstructured:
        self._global_kwargs = kwargs
    self._pruning_method_name = pruning_meth.__name__
    if self._use_global_unstructured:
        return pruning_meth
    return ModelPruning._wrap_pruning_fn(pruning_meth, **kwargs)