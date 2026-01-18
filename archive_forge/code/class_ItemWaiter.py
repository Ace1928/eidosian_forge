import sys
import heapq
import collections
import traceback
from eventlet.event import Event
from eventlet.greenthread import getcurrent
from eventlet.hubs import get_hub
import queue as Stdlib_Queue
from eventlet.timeout import Timeout
class ItemWaiter(Waiter):
    __slots__ = ['item', 'block']

    def __init__(self, item, block):
        Waiter.__init__(self)
        self.item = item
        self.block = block