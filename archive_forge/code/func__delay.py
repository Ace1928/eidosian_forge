import ipaddress
import random
import re
import socket
import time
import weakref
from datetime import timedelta
from threading import Event, Thread
from typing import Any, Callable, Dict, Optional, Tuple, Union
def _delay(seconds: Union[float, Tuple[float, float]]) -> None:
    """Suspend the current thread for ``seconds``.

    Args:
        seconds:
            Either the delay, in seconds, or a tuple of a lower and an upper
            bound within which a random delay will be picked.
    """
    if isinstance(seconds, tuple):
        seconds = random.uniform(*seconds)
    if seconds >= 0.01:
        time.sleep(seconds)