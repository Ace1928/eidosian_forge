from weakref import ref
from contextlib import contextmanager
from threading import current_thread, RLock
Create a new dict for the current thread, and return it.