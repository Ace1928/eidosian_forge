import logging
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from shutil import which
from typing import List, Optional
import torch
from packaging.version import parse
@dataclass
class CPUInformation:
    """
    Stores information about the CPU in a distributed environment. It contains the following attributes:
    - rank: The rank of the current process.
    - world_size: The total number of processes in the world.
    - local_rank: The rank of the current process on the local node.
    - local_world_size: The total number of processes on the local node.
    """
    rank: int = field(default=0, metadata={'help': 'The rank of the current process.'})
    world_size: int = field(default=1, metadata={'help': 'The total number of processes in the world.'})
    local_rank: int = field(default=0, metadata={'help': 'The rank of the current process on the local node.'})
    local_world_size: int = field(default=1, metadata={'help': 'The total number of processes on the local node.'})