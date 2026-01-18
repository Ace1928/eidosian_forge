import bisect
import itertools
import math
from collections import defaultdict, namedtuple
from operator import attrgetter
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.autograd import DeviceType
def _attr_formatter(name):
    return property(lambda self: _format_time(getattr(self, name)))