from typing import Optional
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.utilities.compute import _adjust_weights_safe_divide, _safe_divide
from torchmetrics.utilities.enums import ClassificationTask
def _multilabel_fbeta_score_arg_validation(beta: float, num_labels: int, threshold: float=0.5, average: Optional[Literal['micro', 'macro', 'weighted', 'none']]='macro', multidim_average: Literal['global', 'samplewise']='global', ignore_index: Optional[int]=None) -> None:
    if not (isinstance(beta, float) and beta > 0):
        raise ValueError(f'Expected argument `beta` to be a float larger than 0, but got {beta}.')
    _multilabel_stat_scores_arg_validation(num_labels, threshold, average, multidim_average, ignore_index)