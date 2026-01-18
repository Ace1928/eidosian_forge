from collections import OrderedDict
from typing import Dict, List, Tuple
import torch
from lightning_utilities.core.imports import RequirementCache
from torch.nn import Parameter
from typing_extensions import override
from pytorch_lightning.utilities.model_summary.model_summary import (
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