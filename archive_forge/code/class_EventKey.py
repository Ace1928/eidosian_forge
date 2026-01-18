import functools
import re
from collections import deque
from dataclasses import dataclass
from typing import Dict, List
from torch.autograd import _KinetoEvent
from torch.autograd.profiler import profile
from torch.profiler import DeviceType
class EventKey:

    def __init__(self, event):
        self.event = event

    def __hash__(self):
        return hash(self.event.id)

    def __eq__(self, other):
        return self.event.id == other.event.id

    def __repr__(self):
        return f'{self.event.name}'

    def intervals_overlap(self, intervals: List[Interval]):
        overlap_time = 0
        intervals = sorted(intervals, key=lambda x: x.start)
        if intervals:
            overlap_start = max(self.event.start_time_ns, intervals[0].start)
            overlap_end = min(self.event.end_time_ns, intervals[0].end)
            if overlap_start < overlap_end:
                overlap_time += overlap_end - overlap_start
        i, j = (0, 1)
        while j < len(intervals):
            prev_interval = intervals[i]
            curr_interval = intervals[j]
            j += 1
            if prev_interval.end > curr_interval.start:
                if prev_interval.end > curr_interval.end:
                    j += 1
                    continue
                else:
                    curr_interval.start = prev_interval.end
                    i = j
            overlap_start = max(self.event.start_time_ns, curr_interval.start)
            overlap_end = min(self.event.end_time_ns, curr_interval.end)
            if overlap_start < overlap_end:
                overlap_time += overlap_end - overlap_start
        return overlap_time