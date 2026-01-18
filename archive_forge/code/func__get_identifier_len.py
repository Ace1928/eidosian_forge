import functools
import logging
import sys
import time
import traceback
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.modeling import NOTSET as _NotSpecified
def _get_identifier_len(self):
    stage_timers = list(self.timers.items())
    stage_lengths = list()
    while len(stage_timers) > 0:
        new_stage_timers = list()
        max_len = 0
        for identifier, timer in stage_timers:
            new_stage_timers.extend(timer.timers.items())
            if len(identifier) > max_len:
                max_len = len(identifier)
        stage_lengths.append(max(max_len, len('other')))
        stage_timers = new_stage_timers
    return stage_lengths