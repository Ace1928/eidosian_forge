import logging
from collections import abc, defaultdict
from typing import Any, Dict, Iterable, List, Optional, overload, Sequence, Tuple, Union
import torch
import torch.distributed as dist
from torch.cuda.amp.grad_scaler import _MultiDeviceReplicator, GradScaler, OptState
from torch.distributed.distributed_c10d import ProcessGroup
def _refresh_per_optimizer_state() -> Dict[str, Any]:
    return {'stage': OptState.READY, 'found_inf_per_device': {}}