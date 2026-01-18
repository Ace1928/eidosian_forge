from collections import namedtuple
from typing import Any, Dict, List, Optional, Union
import torch
from torch.distributed import ProcessGroup
from vllm.model_executor.parallel_utils import cupy_utils
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.parallel_utils.custom_all_reduce import custom_all_reduce
Broadcast the input tensor dictionary.