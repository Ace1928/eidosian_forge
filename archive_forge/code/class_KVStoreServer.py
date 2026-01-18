import ctypes
import sys
import pickle
import logging
from ..base import _LIB, check_call
from .base import create
class KVStoreServer(object):
    """The key-value store server."""

    def __init__(self, kvstore):
        """Initialize a new KVStoreServer.

        Parameters
        ----------
        kvstore : KVStore
        """
        self.kvstore = kvstore
        self.handle = kvstore.handle
        self.init_logginig = False

    def _controller(self):
        """Return the server controller."""

        def server_controller(cmd_id, cmd_body, _):
            """Server controler."""
            if not self.init_logginig:
                head = '%(asctime)-15s Server[' + str(self.kvstore.rank) + '] %(message)s'
                logging.basicConfig(level=logging.DEBUG, format=head)
                self.init_logginig = True
            if cmd_id == 0:
                try:
                    optimizer = pickle.loads(cmd_body)
                except:
                    raise
                self.kvstore.set_optimizer(optimizer)
            else:
                print('server %d, unknown command (%d, %s)' % (self.kvstore.rank, cmd_id, cmd_body))
        return server_controller

    def run(self):
        """Run the server, whose behavior is like.


        >>> while receive(x):
        ...     if is_command x: controller(x)
        ...     else if is_key_value x: updater(x)
        """
        _ctrl_proto = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
        check_call(_LIB.MXKVStoreRunServer(self.handle, _ctrl_proto(self._controller()), None))