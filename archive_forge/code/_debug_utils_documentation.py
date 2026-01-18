import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum
from typing import Dict, Iterator, List, Set, Tuple
import torch
import torch.distributed.fsdp._flat_param as flat_param_file
from torch.distributed.fsdp._common_utils import (

    It is used for composable fully_shard() code path, it returns
      1. sharded module tree info: each line reprents a submodule name that contats the
    submodule's FQN and its submodule class name, if the submodule is sharded by `fully_shard`,
    the submodule name will add a postfix with ' FULLY SHARDED'. Each increased tree
    level adds 4 spaces before the printed name. A printed sharded module tree info for a toy model
    is like this:
        [CompositeModel] FULLY SHARDED
            l1[Linear]
            u1[UnitModule] FULLY SHARDED
                u1.l1[Linear]
                u1.seq[Sequential]
                    u1.seq.0[ReLU]
                    u1.seq.1[Linear]
                    u1.seq.2[ReLU]
                u1.l2[Linear]
            u2[UnitModule] FULLY SHARDED
                u2.l1[Linear]
                u2.seq[Sequential]
                    u2.seq.0[ReLU]
                    u2.seq.1[Linear]
                    u2.seq.2[ReLU]
                u2.l2[Linear]
            l2[Linear]
      2. a dict mapping from the concated module FQN and class name to a list of its managed
    original parameters' FQNs. An example of the dict for the above toy sharded model is like this:
            {'[CompositeModel]': ['l1.weight', 'l1.bias', 'l2.weight', 'l2.bias'],
             'u1[UnitModule]': ['u1.l1.weight', 'u1.l1.bias', 'u1.seq.1.weight', 'u1.seq.1.bias', 'u1.l2.weight', 'u1.l2.bias'],
             'u2[UnitModule]': ['u2.l1.weight', 'u2.l1.bias', 'u2.seq.1.weight', 'u2.seq.1.bias', 'u2.l2.weight', 'u2.l2.bias']
            }
    All FQNs are prefixed starting from ``model``.

    Args:
        model (torch.nn.Module): Root module (which may or may not be passed to
                                 composable `fully_shard()`).
    