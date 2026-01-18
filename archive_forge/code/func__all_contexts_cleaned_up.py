import sys
import threading
import time
from enum import Enum
import random
import torch
import torch.nn as nn
from datetime import timedelta
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.testing._internal.dist_utils
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributed.rpc import RRef
from torch.testing._internal.common_utils import IS_MACOS, skip_but_pass_in_sandcastle_if
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
def _all_contexts_cleaned_up(timeout_seconds=10):
    global known_context_ids
    start = time.time()
    context_id_to_raised = set()
    while time.time() - start < timeout_seconds and context_id_to_raised != known_context_ids:
        for context_id in known_context_ids:
            try:
                dist_autograd._retrieve_context(context_id)
            except RuntimeError:
                context_id_to_raised.add(context_id)
    success = context_id_to_raised == known_context_ids
    return success