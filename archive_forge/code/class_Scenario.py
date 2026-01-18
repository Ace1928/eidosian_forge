import argparse
import contextlib
import dataclasses
import enum
import multiprocessing
import os
import random
from collections import deque
from statistics import mean, stdev
from typing import Callable
import torch
@dataclasses.dataclass
class Scenario:
    num_samples: int
    outer_dim: int
    inner_dim: int
    num_ag_matrices: int