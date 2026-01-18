from collections import defaultdict, namedtuple
import gc
import os
import re
import time
import tracemalloc
from typing import Callable, List, Optional
from ray.util.annotations import DeveloperAPI
def _i_print(i):
    if (i + 1) % 10 == 0:
        print('.', end='' if (i + 1) % 100 else f' {i + 1}\n', flush=True)