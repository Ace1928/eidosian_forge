import asyncio
import functools
import queue
import threading
import time
from typing import (
def _clamp(x: float, low: float, high: float) -> float:
    return max(low, min(x, high))