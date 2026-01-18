import dill
import random
import signal
import anyio
from aiocache import cached
from aiocache.serializers import PickleSerializer, BaseSerializer
from anyio.abc import CancelScope
from typing import List, Union, Callable, Any
from lazyops.utils.logs import logger
class InfiniteBackoffsWithJitter:

    def __iter__(self):
        while True:
            yield (10 + random.randint(-5, +5))