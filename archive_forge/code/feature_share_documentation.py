from functools import lru_cache
from typing import Any, Dict, Optional, Sequence, Union
from torch.nn import Module
from torchmetrics.collections import MetricCollection
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
Call the network with the given arguments.