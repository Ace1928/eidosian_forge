from enum import Enum
from queue import Queue
from threading import Thread
from typing import Callable, Optional, List
from .errors import AsyncTaskException
class WaitType(str, Enum):
    ALL = 'all'
    FIRST = 'first'