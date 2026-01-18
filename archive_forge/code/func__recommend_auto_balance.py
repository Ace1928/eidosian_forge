from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast
import torch
from torch import Tensor, nn
from torch.distributed.rpc import RRef
import torch.autograd
import torch.cuda
from . import microbatch
from .batchnorm import DeferredBatchNorm
from .pipeline import Pipeline
from .skip.layout import inspect_skip_layout
from .skip.skippable import verify_skippables
from .stream import AbstractStream, new_stream
def _recommend_auto_balance(message: str) -> str:
    """Expands a message with recommendation to :mod:`torchpipe.balance`."""
    return f"{message}\n\nIf your model is still under development, its optimal balance would change\nfrequently. In this case, we highly recommend 'torch.distributed.pipeline.sync.balance' for\nnaive automatic balancing:\n\n  from torch.distributed.pipeline.sync import Pipe\n  from torch.distributed.pipeline.sync.balance import balance_by_time\n\n  partitions = torch.cuda.device_count()\n  sample = torch.empty(...)\n  balance = balance_by_time(partitions, model, sample)\n\n  model = Pipe(model, balance, ...)\n"