from __future__ import annotations
import collections
import functools
import logging
import math
import os
import threading
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from typing import (
class CacheInfo(NamedTuple):
    hits: int
    misses: int
    maxsize: int
    currsize: int