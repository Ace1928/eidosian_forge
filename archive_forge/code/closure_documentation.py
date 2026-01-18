import os
import threading
from queue import Empty as EmptyQueue, Queue
from torch._lazy.device_context import get_device_context
Start closure event loop if not started