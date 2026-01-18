import bisect
import itertools
import math
from collections import defaultdict, namedtuple
from operator import attrgetter
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.autograd import DeviceType
class MemRecordsAcc:
    """Acceleration structure for accessing mem_records in interval."""

    def __init__(self, mem_records):
        self._mem_records = mem_records
        self._start_uses: List[int] = []
        self._indices: List[int] = []
        if len(mem_records) > 0:
            tmp = sorted([(r[0].start_us(), i) for i, r in enumerate(mem_records)])
            self._start_uses, self._indices = zip(*tmp)

    def in_interval(self, start_us, end_us):
        start_idx = bisect.bisect_left(self._start_uses, start_us)
        end_idx = bisect.bisect_right(self._start_uses, end_us)
        for i in range(start_idx, end_idx):
            yield self._mem_records[self._indices[i]]