import logging
import numbers
from typing import Any, Callable, List, Optional, Tuple
from ray._private.dict import flatten_dict
from ray.air._internal.util import is_nan
from ray.air.config import MAX
from ray.train import CheckpointConfig
from ray.train._internal.session import _TrainingResult
from ray.train._internal.storage import _delete_fs_path
def _get_checkpoint_score(self, checkpoint: _TrainingResult) -> Tuple[bool, numbers.Number]:
    """Get the score for a checkpoint, according to checkpoint config.

        If `mode="min"`, the metric is negated so that the lowest score is
        treated as the best.

        Returns:
            Tuple: A tuple of (not_is_nan: bool, score: numbers.Number).
                This score orders: nan values < float("-inf") < valid numeric metrics
        """
    checkpoint_score_attribute = self._checkpoint_config.checkpoint_score_attribute
    if checkpoint_score_attribute:
        flat_metrics = flatten_dict(checkpoint.metrics)
        try:
            checkpoint_result = flat_metrics[checkpoint_score_attribute]
        except KeyError:
            valid_keys = list(flat_metrics.keys())
            logger.error(f'Result dict has no key: {checkpoint_score_attribute}. checkpoint_score_attr must be set to a key in the result dict. Valid keys are: {valid_keys}')
            checkpoint_result = float('-inf')
    else:
        checkpoint_result = float('-inf')
    checkpoint_score_order = self._checkpoint_config.checkpoint_score_order
    order_factor = 1.0 if checkpoint_score_order == MAX else -1.0
    checkpoint_score = order_factor * checkpoint_result
    if not isinstance(checkpoint_score, numbers.Number):
        raise ValueError(f'Unable to persist checkpoint for checkpoint_score_attribute: {checkpoint_score_attribute} with value {checkpoint_score}. This attribute must be numerical.')
    return (not is_nan(checkpoint_score), checkpoint_score) if not is_nan(checkpoint_score) else (False, float('-inf'))