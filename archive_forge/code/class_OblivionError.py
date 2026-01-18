from _collections import deque
from collections import defaultdict
from functools import total_ordering
from typing import Any, Set, Dict, Union, NewType, Mapping, Tuple, Iterable
from interegular.utils import soft_repr
class OblivionError(Exception):
    """
        This exception is thrown while `crawl()`ing an FSM if we transition to the
        oblivion state. For example while crawling two FSMs in parallel we may
        transition to the oblivion state of both FSMs at once. This warrants an
        out-of-bound signal which will reduce the complexity of the new FSM's map.
    """
    pass