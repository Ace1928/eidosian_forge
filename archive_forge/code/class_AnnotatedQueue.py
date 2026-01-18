import copy
from collections import OrderedDict
from contextlib import contextmanager
from threading import RLock
from typing import Optional
class AnnotatedQueue(OrderedDict):
    """Lightweight class that maintains a basic queue of operations, in addition
    to metadata annotations."""
    _lock = RLock()
    'threading.RLock: Used to synchronize appending to/popping from global QueueingContext.'

    def __enter__(self):
        """Adds this instance to the global list of active contexts.

        Returns:
            AnnotatedQueue: this instance
        """
        AnnotatedQueue._lock.acquire()
        QueuingManager.add_active_queue(self)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Remove this instance from the global list of active contexts."""
        QueuingManager.remove_active_queue()
        AnnotatedQueue._lock.release()

    def append(self, obj, **kwargs):
        """Append ``obj`` into the queue with ``kwargs`` metadata."""
        obj = obj if isinstance(obj, WrappedObj) else WrappedObj(obj)
        self[obj] = kwargs

    def remove(self, obj):
        """Remove ``obj`` from the queue. Passes silently if the object is not in the queue."""
        obj = obj if isinstance(obj, WrappedObj) else WrappedObj(obj)
        if obj in self:
            del self[obj]

    def update_info(self, obj, **kwargs):
        """Update ``obj``'s metadata with ``kwargs`` if it exists in the queue."""
        obj = obj if isinstance(obj, WrappedObj) else WrappedObj(obj)
        if obj in self:
            self[obj].update(kwargs)

    def get_info(self, obj):
        """Retrieve the metadata for ``obj``.  Raises a ``QueuingError`` if obj is not in the queue."""
        obj = obj if isinstance(obj, WrappedObj) else WrappedObj(obj)
        if obj not in self:
            raise QueuingError(f'Object {obj.obj} not in the queue.')
        return self[obj]

    def items(self):
        return tuple(((key.obj, value) for key, value in super().items()))

    @property
    def queue(self):
        """Returns a list of objects in the annotated queue"""
        return list((key.obj for key in self.keys()))

    def __setitem__(self, key, value):
        key = key if isinstance(key, WrappedObj) else WrappedObj(key)
        return super().__setitem__(key, value)

    def __getitem__(self, key):
        key = key if isinstance(key, WrappedObj) else WrappedObj(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        key = key if isinstance(key, WrappedObj) else WrappedObj(key)
        return super().__contains__(key)