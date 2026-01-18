import asyncio
import functools
import hashlib
import os
from typing import Callable, Optional
import cloudpickle
from diskcache import Cache
Erase the cache completely.