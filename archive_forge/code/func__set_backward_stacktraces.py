import bisect
import itertools
import math
from collections import defaultdict, namedtuple
from operator import attrgetter
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.autograd import DeviceType
def _set_backward_stacktraces(self):

    def bw_parent(evt):
        if evt is None:
            return None
        elif evt.scope == 1:
            return evt
        else:
            return bw_parent(evt.cpu_parent)
    fwd_stacks = {}
    for evt in self:
        if bw_parent(evt) is None and evt.stack is not None:
            t = (evt.sequence_nr, evt.thread)
            if t not in fwd_stacks:
                fwd_stacks[t] = evt.stack
    for evt in self:
        p = bw_parent(evt)
        if p is not None:
            assert p.fwd_thread is not None
            t = (p.sequence_nr, p.fwd_thread)
            if t in fwd_stacks:
                evt.stack = fwd_stacks[t]
            else:
                evt.stack = []