import torch
import random
import os
import queue
from dataclasses import dataclass
from torch._utils import ExceptionWrapper
from typing import Optional, Union, TYPE_CHECKING
from . import signal_handling, MP_STATUS_CHECK_INTERVAL, IS_WINDOWS, HAS_NUMPY
class WorkerInfo:
    id: int
    num_workers: int
    seed: int
    dataset: 'Dataset'
    __initialized = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.__keys = tuple(kwargs.keys())
        self.__initialized = True

    def __setattr__(self, key, val):
        if self.__initialized:
            raise RuntimeError(f'Cannot assign attributes to {self.__class__.__name__} objects')
        return super().__setattr__(key, val)

    def __repr__(self):
        items = []
        for k in self.__keys:
            items.append(f'{k}={getattr(self, k)}')
        return f'{self.__class__.__name__}({', '.join(items)})'