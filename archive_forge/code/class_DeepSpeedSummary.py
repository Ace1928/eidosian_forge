from collections import OrderedDict
from typing import Dict, List, Tuple
import torch
from lightning_utilities.core.imports import RequirementCache
from torch.nn import Parameter
from typing_extensions import override
from pytorch_lightning.utilities.model_summary.model_summary import (
class DeepSpeedSummary(ModelSummary):

    @override
    def summarize(self) -> Dict[str, DeepSpeedLayerSummary]:
        summary = OrderedDict(((name, DeepSpeedLayerSummary(module)) for name, module in self.named_modules))
        if self._model.example_input_array is not None:
            self._forward_example_input()
        for layer in summary.values():
            layer.detach_hook()
        if self._max_depth >= 1:
            for k in [k for k in summary if k.count('.') >= self._max_depth]:
                del summary[k]
        return summary

    @property
    @override
    def total_parameters(self) -> int:
        return sum((deepspeed_param_size(p) if not _is_lazy_weight_tensor(p) else 0 for p in self._model.parameters()))

    @property
    @override
    def trainable_parameters(self) -> int:
        return sum((deepspeed_param_size(p) if not _is_lazy_weight_tensor(p) else 0 for p in self._model.parameters() if p.requires_grad))

    @property
    def parameters_per_layer(self) -> List[int]:
        return [layer.average_shard_parameters for layer in self._layer_summary.values()]

    @override
    def _get_summary_data(self) -> List[Tuple[str, List[str]]]:
        """Makes a summary listing with:

        Layer Name, Layer Type, Number of Parameters, Input Sizes, Output Sizes, Model Size

        """
        arrays = [(' ', list(map(str, range(len(self._layer_summary))))), ('Name', self.layer_names), ('Type', self.layer_types), ('Params', list(map(get_human_readable_count, self.param_nums))), ('Params per Device', list(map(get_human_readable_count, self.parameters_per_layer)))]
        if self._model.example_input_array is not None:
            arrays.append(('In sizes', [str(x) for x in self.in_sizes]))
            arrays.append(('Out sizes', [str(x) for x in self.out_sizes]))
        total_leftover_params = self.total_parameters - self.total_layer_params
        if total_leftover_params > 0:
            self._add_leftover_params_to_summary(arrays, total_leftover_params)
        return arrays

    @override
    def _add_leftover_params_to_summary(self, arrays: List[Tuple[str, List[str]]], total_leftover_params: int) -> None:
        """Add summary of params not associated with module or layer to model summary."""
        super()._add_leftover_params_to_summary(arrays, total_leftover_params)
        layer_summaries = dict(arrays)
        layer_summaries['Params per Device'].append(NOT_APPLICABLE)