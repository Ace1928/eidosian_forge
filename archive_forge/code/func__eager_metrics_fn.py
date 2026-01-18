import numpy as np
from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.mixed_precision import loss_scale_optimizer
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
def _eager_metrics_fn(model, outputs, targets, sample_weights=None, masks=None):
    """Calculates the metrics for each output of the given model.

  Args:
      model: The model on which metrics are being calculated.
      outputs: The outputs of the given model.
      targets: The predictions or targets of the given model.
      sample_weights: Optional list of sample weights for each output.
      masks: Optional list of masks for each output.

  Returns:
      Returns the metric results for each output of the model.
  """
    outputs = nest.flatten(outputs)
    targets = nest.flatten(targets)
    metric_results = []
    if targets:
        if len(model._targets) != len(targets):
            new_targets = [None if t is None else targets.pop(0) for t in model._targets]
            targets = new_targets
        metric_results = model._handle_metrics(outputs, targets=targets, sample_weights=sample_weights, masks=masks, return_weighted_and_unweighted_metrics=True, skip_target_masks=model._prepare_skip_target_masks())
    metric_results.extend([m.result() for m in model.metrics if m not in model._compile_metric_functions])
    return metric_results