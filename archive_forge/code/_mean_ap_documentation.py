import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.distributed as dist
from torch import IntTensor, Tensor
from torchmetrics.detection.helpers import _fix_empty_tensors, _input_validator
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import _cumsum
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _PYCOCOTOOLS_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_8
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure object and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> from torch import tensor
            >>> from torchmetrics.detection.mean_ap import MeanAveragePrecision
            >>> preds = [dict(
            ...     boxes=tensor([[258.0, 41.0, 606.0, 285.0]]),
            ...     scores=tensor([0.536]),
            ...     labels=tensor([0]),
            ... )]
            >>> target = [dict(
            ...     boxes=tensor([[214.0, 41.0, 562.0, 285.0]]),
            ...     labels=tensor([0]),
            ... )]
            >>> metric = MeanAveragePrecision()
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.detection.mean_ap import MeanAveragePrecision
            >>> preds = lambda: [dict(
            ...     boxes=torch.tensor([[258.0, 41.0, 606.0, 285.0]]) + torch.randint(10, (1,4)),
            ...     scores=torch.tensor([0.536]) + 0.1*torch.rand(1),
            ...     labels=torch.tensor([0]),
            ... )]
            >>> target = [dict(
            ...     boxes=torch.tensor([[214.0, 41.0, 562.0, 285.0]]),
            ...     labels=torch.tensor([0]),
            ... )]
            >>> metric = MeanAveragePrecision()
            >>> vals = []
            >>> for _ in range(20):
            ...     vals.append(metric(preds(), target))
            >>> fig_, ax_ = metric.plot(vals)

        