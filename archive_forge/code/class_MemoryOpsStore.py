import abc
import os
class MemoryOpsStore:
    """ A multi-op store that stores the operations in memory.
    """

    def __init__(self):
        self._store = {}

    def put_ops(self, key, time, ops):
        """ Put an ops only if not already there, otherwise it's a no op.
        """
        if self._store.get(key) is None:
            self._store[key] = ops

    def get_ops(self, key):
        """ Returns ops from the key if found otherwise raises a KeyError.
        """
        ops = self._store.get(key)
        if ops is None:
            raise KeyError('cannot get operations for {}'.format(key))
        return ops