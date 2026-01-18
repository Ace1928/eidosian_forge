import copy
import functools
import gc
import inspect
import os
import random
import re
import threading
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import numpy as np
from .utils import (
class SchedulerType(ExplicitEnum):
    LINEAR = 'linear'
    COSINE = 'cosine'
    COSINE_WITH_RESTARTS = 'cosine_with_restarts'
    POLYNOMIAL = 'polynomial'
    CONSTANT = 'constant'
    CONSTANT_WITH_WARMUP = 'constant_with_warmup'
    INVERSE_SQRT = 'inverse_sqrt'
    REDUCE_ON_PLATEAU = 'reduce_lr_on_plateau'