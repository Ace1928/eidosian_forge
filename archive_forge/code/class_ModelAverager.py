import warnings
from abc import ABC, abstractmethod
from typing import Union, Iterable, Dict
import torch
import torch.distributed as dist
import torch.distributed.algorithms.model_averaging.utils as utils
class ModelAverager(ABC):
    """Base class for all model averagers.

    Args:
        process_group: The process group to be used for all-reduce.
                       If ``None``, the default process group, which
                       is created by :func:`torch.distributed.init_process_group`,
                       will be used. (default: ``None``)
    """

    def __init__(self, process_group=None):
        self.process_group = process_group if process_group is not None else dist.group.WORLD
        self.step = 0

    @abstractmethod
    def average_parameters(self, params):
        raise NotImplementedError