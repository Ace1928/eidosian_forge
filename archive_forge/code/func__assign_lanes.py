import collections
import copy
import json
import re
from tensorflow.python.platform import build_info
from tensorflow.python.platform import tf_logging as logging
def _assign_lanes(self):
    """Assigns non-overlapping lanes for the activities on each device."""
    for device_stats in self._step_stats.dev_stats:
        lanes = [0]
        for ns in device_stats.node_stats:
            l = -1
            for i, lts in enumerate(lanes):
                if ns.all_start_micros > lts:
                    l = i
                    lanes[l] = ns.all_start_micros + ns.all_end_rel_micros
                    break
            if l < 0:
                l = len(lanes)
                lanes.append(ns.all_start_micros + ns.all_end_rel_micros)
            ns.thread_id = l