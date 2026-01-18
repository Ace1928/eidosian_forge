import collections
import csv
import re
from typing import (Any, Callable, Dict, IO, Iterable, List, Mapping, Optional,
import numpy as np
from tensorflow.lite.python import convert
from tensorflow.lite.python import interpreter as _interpreter
from tensorflow.lite.python.metrics import metrics as metrics_stub  # type: ignore
from tensorflow.python.util import tf_export
def _collect_model_statistics(self) -> Dict[str, float]:
    """Collects model output metrics.

    For all data from the given RepresentativeDataset, collect all model output
    results from float model & quantized debug model, and calculate metrics
    by using model output functions. As a result, self.model_results is filled,

    where self.model_results[model_output_function_name] = `aggregated model
    output function value` (a scalar).

    Returns:
      aggregated per-model output discrepancy metrics.
      {metric_name: aggregated_metric}
    """
    model_statistics = collections.defaultdict(list)
    initialize = True
    for tensor_data in self._data_gen():
        self._set_input_tensors(self._quant_interpreter, tensor_data, initialize)
        self._quant_interpreter.invoke()
        quant_tensor_data = self._get_output_tensors(self._quant_interpreter)
        float_tensor_data = []
        if self._float_interpreter:
            self._set_input_tensors(self._float_interpreter, tensor_data, initialize)
            self._float_interpreter.invoke()
            float_tensor_data = self._get_output_tensors(self._float_interpreter)
        initialize = False
        for metric_name, metric_fn in self._debug_options.model_debug_metrics.items():
            model_statistics[metric_name].append(metric_fn(float_tensor_data, quant_tensor_data))
    return {metric_name: np.mean(metric) for metric_name, metric in model_statistics.items()}