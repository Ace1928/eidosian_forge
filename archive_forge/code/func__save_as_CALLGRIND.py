import os
import sys
import _yappi
import pickle
import threading
import warnings
import types
import inspect
import itertools
from contextlib import contextmanager
def _save_as_CALLGRIND(self, path):
    """
        Writes all the function stats in a callgrind-style format to the given
        file. (stdout by default)
        """
    header = 'version: 1\ncreator: %s\npid: %d\ncmd:  %s\npart: 1\n\nevents: Ticks' % ('yappi', os.getpid(), ' '.join(sys.argv))
    lines = [header]
    file_ids = ['']
    func_ids = ['']
    func_idx_list = []
    for func_stat in self:
        file_ids += ['fl=(%d) %s' % (func_stat.index, func_stat.module)]
        func_ids += ['fn=(%d) %s %s:%s' % (func_stat.index, func_stat.name, func_stat.module, func_stat.lineno)]
        func_idx_list.append(func_stat.index)
        for child in func_stat.children:
            if child.index in func_idx_list:
                continue
            file_ids += ['fl=(%d) %s' % (child.index, child.module)]
            func_ids += ['fn=(%d) %s %s:%s' % (child.index, child.name, child.module, child.lineno)]
            func_idx_list.append(child.index)
    lines += file_ids + func_ids
    for func_stat in self:
        func_stats = ['', 'fl=(%d)' % func_stat.index, 'fn=(%d)' % func_stat.index]
        func_stats += [f'{func_stat.lineno} {int(func_stat.tsub * 1000000.0)}']
        for child in func_stat.children:
            func_stats += ['cfl=(%d)' % child.index, 'cfn=(%d)' % child.index, 'calls=%d 0' % child.ncall, '0 %d' % int(child.ttot * 1000000.0)]
        lines += func_stats
    with open(path, 'w') as f:
        f.write('\n'.join(lines))