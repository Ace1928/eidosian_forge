import bisect
import itertools
import math
from collections import defaultdict, namedtuple
from operator import attrgetter
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.autograd import DeviceType
def _populate_cpu_children(self):
    """Populate child events into each underlying FunctionEvent object.

        One event is a child of another if [s1, e1) is inside [s2, e2). Where
        s1 and e1 would be start and end of the child event's interval. And
        s2 and e2 start and end of the parent event's interval

        Example: In event list [[0, 10], [1, 3], [3, 4]] would have make [0, 10]
        be a parent of two other intervals.

        If for any reason two intervals intersect only partially, this function
        will not record a parent child relationship between then.
        """
    sync_events = [evt for evt in self if not evt.is_async and evt.device_type == DeviceType.CPU]
    events = sorted(sync_events, key=attrgetter('thread'))
    threads = itertools.groupby(events, key=lambda event: (event.thread, event.node_id))
    for thread_id, thread_events in threads:
        thread_events_ = sorted(thread_events, key=lambda event: [event.time_range.start, -event.time_range.end])
        current_events: List[FunctionEvent] = []
        cur_end = 0
        for event in thread_events_:
            while len(current_events) > 0:
                parent = current_events[-1]
                if event.time_range.start >= parent.time_range.end or event.time_range.end > parent.time_range.end:
                    current_events.pop()
                else:
                    parent.append_cpu_child(event)
                    assert event.cpu_parent is None, f'There is already a CPU parent event for {event.key}'
                    event.set_cpu_parent(parent)
                    break
            current_events.append(event)