import functools
import logging
import sys
import time
import traceback
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.modeling import NOTSET as _NotSpecified
class _HierarchicalHelper(object):

    def __init__(self):
        self.tic_toc = TicTocTimer()
        self.timers = dict()
        self.total_time = 0
        self.n_calls = 0

    def start(self):
        self.n_calls += 1
        self.tic_toc.start()

    def stop(self):
        self.total_time += self.tic_toc.stop()

    def to_str(self, indent, stage_identifier_lengths):
        s = ''
        if len(self.timers) > 0:
            underline = indent + '-' * (sum(stage_identifier_lengths) + 36) + '\n'
            s += underline
            name_formatter = '{name:<' + str(sum(stage_identifier_lengths)) + '}'
            other_time = self.total_time
            sub_stage_identifier_lengths = stage_identifier_lengths[1:]
            for name, timer in sorted(self.timers.items()):
                if self.total_time > 0:
                    _percent = timer.total_time / self.total_time * 100
                else:
                    _percent = float('nan')
                s += indent
                s += (name_formatter + '{ncalls:>9d} {cumtime:>9.3f} {percall:>9.3f} {percent:>6.1f}\n').format(name=name, ncalls=timer.n_calls, cumtime=timer.total_time, percall=timer.total_time / timer.n_calls, percent=_percent)
                s += timer.to_str(indent=indent + ' ' * stage_identifier_lengths[0], stage_identifier_lengths=sub_stage_identifier_lengths)
                other_time -= timer.total_time
            if self.total_time > 0:
                _percent = other_time / self.total_time * 100
            else:
                _percent = float('nan')
            s += indent
            s += (name_formatter + '{ncalls:>9} {cumtime:>9.3f} {percall:>9} {percent:>6.1f}\n').format(name='other', ncalls='n/a', cumtime=other_time, percall='n/a', percent=_percent)
            s += underline.replace('-', '=')
        return s

    def get_timers(self, res, prefix):
        for name, timer in self.timers.items():
            _name = prefix + '.' + name
            res.append(_name)
            timer.get_timers(res, _name)

    def flatten(self):
        items = list(self.timers.items())
        for child_key, child_timer in items:
            child_timer.flatten()
            _move_grandchildren_to_root(self, child_timer)

    def clear_except(self, *args):
        to_retain = set(args)
        _clear_timers_except(self, to_retain)