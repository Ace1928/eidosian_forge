from collections import defaultdict, namedtuple
import gc
import os
import re
import time
import tracemalloc
from typing import Callable, List, Optional
from ray.util.annotations import DeveloperAPI
def _find_memory_leaks_in_table(table):
    import scipy.stats
    import numpy as np
    suspects = []
    for traceback, hist in table.items():
        memory_increase = hist[-1] - hist[0]
        if memory_increase <= 0.0:
            continue
        top_stack = str(traceback[-1])
        drive_separator = '\\\\' if os.name == 'nt' else '/'
        if any((s in top_stack for s in ['tracemalloc', 'pycharm', 'thirdparty_files/psutil', re.sub('\\.', drive_separator, __name__) + '.py'])):
            continue
        line = scipy.stats.linregress(x=np.arange(len(hist)), y=np.array(hist))
        if memory_increase > 1000 and (line.slope > 60.0 and line.rvalue > 0.875 or (line.slope > 20.0 and line.rvalue > 0.9) or (line.slope > 10.0 and line.rvalue > 0.95)):
            suspects.append(Suspect(traceback=traceback, memory_increase=memory_increase, slope=line.slope, rvalue=line.rvalue, hist=hist))
    return suspects