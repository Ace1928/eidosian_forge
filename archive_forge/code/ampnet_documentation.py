import time
from typing import Any, Dict, List, Tuple, Union
import torch
from torch import nn
from torch.autograd.profiler import record_function
from torch.distributed import ProcessGroup
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from fairscale.nn.model_parallel import get_pipeline_parallel_ranks
from fairscale.nn.pipe.async_schedule import (
from fairscale.nn.pipe.checkpoint import Checkpointing
from fairscale.nn.pipe.messages import Transport
from fairscale.nn.pipe.microbatch import Batch
from fairscale.nn.pipe.types import (
from fairscale.nn.pipe.worker import Task
Get the tensor(s) wrapped in a `Batch` from a `PipeMessage`, applying
        AsyncRecvOperator so we can intercept the backward pass