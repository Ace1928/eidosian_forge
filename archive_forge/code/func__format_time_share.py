import bisect
import itertools
import math
from collections import defaultdict, namedtuple
from operator import attrgetter
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.autograd import DeviceType
def _format_time_share(time_us, total_time_us):
    """Define how to format time in FunctionEvent."""
    if total_time_us == 0:
        assert time_us == 0, f'Expected time_us == 0 but got {time_us}'
        return 'NaN'
    return f'{time_us * 100.0 / total_time_us:.2f}%'