import re
from contextlib import contextmanager
import functools
import operator
import warnings
import numbers
from collections import namedtuple
import inspect
import math
from typing import (
import numpy as np
from scipy._lib._array_api import array_namespace
def _rng_spawn(rng, n_children):
    bg = rng._bit_generator
    ss = bg._seed_seq
    child_rngs = [np.random.Generator(type(bg)(child_ss)) for child_ss in ss.spawn(n_children)]
    return child_rngs