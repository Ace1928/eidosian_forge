import contextlib
import logging
import math
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle
import pytorch_lightning as pl
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from pytorch_lightning.utilities.model_helpers import _ModuleMode
from pytorch_lightning.utilities.rank_zero import WarningCache
class ModelSummary:
    """Generates a summary of all layers in a :class:`~pytorch_lightning.core.LightningModule`.

    Args:
        model: The model to summarize (also referred to as the root module).

        max_depth: Maximum depth of modules to show. Use -1 to show all modules or 0 to show no
            summary. Defaults to 1.

    The string representation of this summary prints a table with columns containing
    the name, type and number of parameters for each layer.

    The root module may also have an attribute ``example_input_array`` as shown in the example below.
    If present, the root module will be called with it as input to determine the
    intermediate input- and output shapes of all layers. Supported are tensors and
    nested lists and tuples of tensors. All other types of inputs will be skipped and show as `?`
    in the summary table. The summary will also display `?` for layers not used in the forward pass.
    If there are parameters not associated with any layers or modules, the count of those parameters
    will be displayed in the table under `other params`. The summary will display `n/a` for module type,
    in size, and out size.

    Example::

        >>> import pytorch_lightning as pl
        >>> class LitModel(pl.LightningModule):
        ...
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.net = nn.Sequential(nn.Linear(256, 512), nn.BatchNorm1d(512))
        ...         self.example_input_array = torch.zeros(10, 256)  # optional
        ...
        ...     def forward(self, x):
        ...         return self.net(x)
        ...
        >>> model = LitModel()
        >>> ModelSummary(model, max_depth=1)  # doctest: +NORMALIZE_WHITESPACE
          | Name | Type       | Params | In sizes  | Out sizes
        ------------------------------------------------------------
        0 | net  | Sequential | 132 K  | [10, 256] | [10, 512]
        ------------------------------------------------------------
        132 K     Trainable params
        0         Non-trainable params
        132 K     Total params
        0.530     Total estimated model params size (MB)
        >>> ModelSummary(model, max_depth=-1)  # doctest: +NORMALIZE_WHITESPACE
          | Name  | Type        | Params | In sizes  | Out sizes
        --------------------------------------------------------------
        0 | net   | Sequential  | 132 K  | [10, 256] | [10, 512]
        1 | net.0 | Linear      | 131 K  | [10, 256] | [10, 512]
        2 | net.1 | BatchNorm1d | 1.0 K    | [10, 512] | [10, 512]
        --------------------------------------------------------------
        132 K     Trainable params
        0         Non-trainable params
        132 K     Total params
        0.530     Total estimated model params size (MB)

    """

    def __init__(self, model: 'pl.LightningModule', max_depth: int=1) -> None:
        self._model = model
        if not isinstance(max_depth, int) or max_depth < -1:
            raise ValueError(f'`max_depth` can be -1, 0 or > 0, got {max_depth}.')
        self._max_depth = max_depth
        self._layer_summary = self.summarize()
        precision_to_bits = {'64': 64, '32': 32, '16': 16, 'bf16': 16}
        precision = precision_to_bits.get(self._model.trainer.precision, 32) if self._model._trainer else 32
        self._precision_megabytes = precision / 8.0 * 1e-06

    @property
    def named_modules(self) -> List[Tuple[str, nn.Module]]:
        mods: List[Tuple[str, nn.Module]]
        if self._max_depth == 0:
            mods = []
        elif self._max_depth == 1:
            mods = list(self._model.named_children())
        else:
            mods = self._model.named_modules()
            mods = list(mods)[1:]
        return mods

    @property
    def layer_names(self) -> List[str]:
        return list(self._layer_summary.keys())

    @property
    def layer_types(self) -> List[str]:
        return [layer.layer_type for layer in self._layer_summary.values()]

    @property
    def in_sizes(self) -> List:
        return [layer.in_size for layer in self._layer_summary.values()]

    @property
    def out_sizes(self) -> List:
        return [layer.out_size for layer in self._layer_summary.values()]

    @property
    def param_nums(self) -> List[int]:
        return [layer.num_parameters for layer in self._layer_summary.values()]

    @property
    def total_parameters(self) -> int:
        return sum((p.numel() if not _is_lazy_weight_tensor(p) else 0 for p in self._model.parameters()))

    @property
    def trainable_parameters(self) -> int:
        return sum((p.numel() if not _is_lazy_weight_tensor(p) else 0 for p in self._model.parameters() if p.requires_grad))

    @property
    def total_layer_params(self) -> int:
        return sum(self.param_nums)

    @property
    def model_size(self) -> float:
        return self.total_parameters * self._precision_megabytes

    def summarize(self) -> Dict[str, LayerSummary]:
        summary = OrderedDict(((name, LayerSummary(module)) for name, module in self.named_modules))
        if self._model.example_input_array is not None:
            self._forward_example_input()
        for layer in summary.values():
            layer.detach_hook()
        if self._max_depth >= 1:
            for k in [k for k in summary if k.count('.') >= self._max_depth]:
                del summary[k]
        return summary

    def _forward_example_input(self) -> None:
        """Run the example input through each layer to get input- and output sizes."""
        model = self._model
        trainer = self._model._trainer
        input_ = model.example_input_array
        input_ = model._on_before_batch_transfer(input_)
        input_ = model._apply_batch_transfer_handler(input_)
        mode = _ModuleMode()
        mode.capture(model)
        model.eval()
        forward_context = contextlib.nullcontext() if trainer is None else trainer.precision_plugin.forward_context()
        with torch.no_grad(), forward_context:
            if isinstance(input_, (list, tuple)):
                model(*input_)
            elif isinstance(input_, dict):
                model(**input_)
            else:
                model(input_)
        mode.restore(model)

    def _get_summary_data(self) -> List[Tuple[str, List[str]]]:
        """Makes a summary listing with:

        Layer Name, Layer Type, Number of Parameters, Input Sizes, Output Sizes, Model Size

        """
        arrays = [(' ', list(map(str, range(len(self._layer_summary))))), ('Name', self.layer_names), ('Type', self.layer_types), ('Params', list(map(get_human_readable_count, self.param_nums)))]
        if self._model.example_input_array is not None:
            arrays.append(('In sizes', [str(x) for x in self.in_sizes]))
            arrays.append(('Out sizes', [str(x) for x in self.out_sizes]))
        total_leftover_params = self.total_parameters - self.total_layer_params
        if total_leftover_params > 0:
            self._add_leftover_params_to_summary(arrays, total_leftover_params)
        return arrays

    def _add_leftover_params_to_summary(self, arrays: List[Tuple[str, List[str]]], total_leftover_params: int) -> None:
        """Add summary of params not associated with module or layer to model summary."""
        layer_summaries = dict(arrays)
        layer_summaries[' '].append(' ')
        layer_summaries['Name'].append(LEFTOVER_PARAMS_NAME)
        layer_summaries['Type'].append(NOT_APPLICABLE)
        layer_summaries['Params'].append(get_human_readable_count(total_leftover_params))
        if 'In sizes' in layer_summaries:
            layer_summaries['In sizes'].append(NOT_APPLICABLE)
        if 'Out sizes' in layer_summaries:
            layer_summaries['Out sizes'].append(NOT_APPLICABLE)

    def __str__(self) -> str:
        arrays = self._get_summary_data()
        total_parameters = self.total_parameters
        trainable_parameters = self.trainable_parameters
        model_size = self.model_size
        return _format_summary_table(total_parameters, trainable_parameters, model_size, *arrays)

    def __repr__(self) -> str:
        return str(self)