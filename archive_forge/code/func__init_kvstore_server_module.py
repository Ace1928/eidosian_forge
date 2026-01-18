import ctypes
import sys
import pickle
import logging
from ..base import _LIB, check_call
from .base import create
def _init_kvstore_server_module():
    """Start server/scheduler."""
    is_worker = ctypes.c_int()
    check_call(_LIB.MXKVStoreIsWorkerNode(ctypes.byref(is_worker)))
    if is_worker.value == 0:
        kvstore = create('dist')
        server = KVStoreServer(kvstore)
        server.run()
        sys.exit()