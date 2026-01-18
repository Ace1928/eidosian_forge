import time
from multiprocessing import Process
from threading import Thread
from typing import Any, Callable, List, Optional, Tuple
import zmq
from zmq import ENOTSOCK, ETERM, PUSH, QUEUE, Context, ZMQBindError, ZMQError, device
class ThreadDevice(BackgroundDevice):
    """A Device that will be run in a background Thread.

    See Device for details.
    """
    _launch_class = Thread