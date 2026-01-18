import logging
import warnings
import weakref
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from typing import List, Optional, cast
def _outstanding_wait_count() -> int:
    """ Returns the number of outstanding work objects waiting to be waited (sic). """
    return len(data_ptr_to_work)