from typing import Sequence, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
def _explained_variance_compute(num_obs: Union[int, Tensor], sum_error: Tensor, sum_squared_error: Tensor, sum_target: Tensor, sum_squared_target: Tensor, multioutput: Literal['raw_values', 'uniform_average', 'variance_weighted']='uniform_average') -> Tensor:
    """Compute Explained Variance.

    Args:
        num_obs: Number of predictions or observations
        sum_error: Sum of errors over all observations
        sum_squared_error: Sum of square of errors over all observations
        sum_target: Sum of target values
        sum_squared_target: Sum of squares of target values
        multioutput: Defines aggregation in the case of multiple output scores. Can be one
            of the following strings:

            * ``'raw_values'`` returns full set of scores
            * ``'uniform_average'`` scores are uniformly averaged
            * ``'variance_weighted'`` scores are weighted by their individual variances

    Example:
        >>> target = torch.tensor([[0.5, 1], [-1, 1], [7, -6]])
        >>> preds = torch.tensor([[0, 2], [-1, 2], [8, -5]])
        >>> num_obs, sum_error, ss_error, sum_target, ss_target = _explained_variance_update(preds, target)
        >>> _explained_variance_compute(num_obs, sum_error, ss_error, sum_target, ss_target, multioutput='raw_values')
        tensor([0.9677, 1.0000])

    """
    diff_avg = sum_error / num_obs
    numerator = sum_squared_error / num_obs - diff_avg * diff_avg
    target_avg = sum_target / num_obs
    denominator = sum_squared_target / num_obs - target_avg * target_avg
    nonzero_numerator = numerator != 0
    nonzero_denominator = denominator != 0
    valid_score = nonzero_numerator & nonzero_denominator
    output_scores = torch.ones_like(diff_avg)
    output_scores[valid_score] = 1.0 - numerator[valid_score] / denominator[valid_score]
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0
    if multioutput == 'raw_values':
        return output_scores
    if multioutput == 'uniform_average':
        return torch.mean(output_scores)
    denom_sum = torch.sum(denominator)
    return torch.sum(denominator / denom_sum * output_scores)