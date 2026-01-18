from typing import Any, Callable, List, Optional, Tuple, Type, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.enums import ClassificationTask
class _AbstractStatScores(Metric):
    tp: Union[List[Tensor], Tensor]
    fp: Union[List[Tensor], Tensor]
    tn: Union[List[Tensor], Tensor]
    fn: Union[List[Tensor], Tensor]

    def _create_state(self, size: int, multidim_average: Literal['global', 'samplewise']='global') -> None:
        """Initialize the states for the different statistics."""
        default: Union[Callable[[], list], Callable[[], Tensor]]
        if multidim_average == 'samplewise':
            default = list
            dist_reduce_fx = 'cat'
        else:
            default = lambda: torch.zeros(size, dtype=torch.long)
            dist_reduce_fx = 'sum'
        self.add_state('tp', default(), dist_reduce_fx=dist_reduce_fx)
        self.add_state('fp', default(), dist_reduce_fx=dist_reduce_fx)
        self.add_state('tn', default(), dist_reduce_fx=dist_reduce_fx)
        self.add_state('fn', default(), dist_reduce_fx=dist_reduce_fx)

    def _update_state(self, tp: Tensor, fp: Tensor, tn: Tensor, fn: Tensor) -> None:
        """Update states depending on multidim_average argument."""
        if self.multidim_average == 'samplewise':
            self.tp.append(tp)
            self.fp.append(fp)
            self.tn.append(tn)
            self.fn.append(fn)
        else:
            self.tp += tp
            self.fp += fp
            self.tn += tn
            self.fn += fn

    def _final_state(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Aggregate states that are lists and return final states."""
        tp = dim_zero_cat(self.tp)
        fp = dim_zero_cat(self.fp)
        tn = dim_zero_cat(self.tn)
        fn = dim_zero_cat(self.fn)
        return (tp, fp, tn, fn)