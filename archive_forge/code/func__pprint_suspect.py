from collections import defaultdict, namedtuple
import gc
import os
import re
import time
import tracemalloc
from typing import Callable, List, Optional
from ray.util.annotations import DeveloperAPI
def _pprint_suspect(suspect):
    print('Most suspicious memory allocation in traceback (only printing out this one, but all (less suspicious) suspects will be investigated as well):')
    print('\n'.join(suspect.traceback.format()))
    print(f'Increase total={suspect.memory_increase}B')
    print(f'Slope={suspect.slope} B/detection')
    print(f'Rval={suspect.rvalue}')