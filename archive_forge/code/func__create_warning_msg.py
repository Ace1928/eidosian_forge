import functools
import itertools
import logging
import os
import queue
import threading
import warnings
from typing import Any, Callable, Iterable, TypeVar, Generic, List, Optional, Union
import multiprocessing as python_multiprocessing
import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
import torch.utils.data.graph_settings
from torch._utils import ExceptionWrapper
from . import (
from torch.utils.data.datapipes.datapipe import _IterDataPipeSerializationWrapper, _MapDataPipeSerializationWrapper
from . import _utils
def _create_warning_msg(num_worker_suggest, num_worker_created, cpuset_checked):
    suggested_max_worker_msg = 'Our suggested max number of worker in current system is {}{}, which is smaller than what this DataLoader is going to create.'.format(num_worker_suggest, '' if cpuset_checked else ' (`cpuset` is not taken into account)') if num_worker_suggest is not None else 'DataLoader is not able to compute a suggested max number of worker in current system.'
    warn_msg = 'This DataLoader will create {} worker processes in total. {} Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.'.format(num_worker_created, suggested_max_worker_msg)
    return warn_msg