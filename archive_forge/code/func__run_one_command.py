import logging
from multiprocessing import managers
from multiprocessing import util as mp_util
import threading
import time
import weakref
import oslo_rootwrap
from oslo_rootwrap import daemon
from oslo_rootwrap import jsonrpc
from oslo_rootwrap import subprocess
def _run_one_command(self, proxy, cmd, stdin):
    """Wrap proxy.run_one_command, setting _need_restart on an exception.

        Usually it should be enough to drain stale data on socket
        rather than to restart, but we cannot do draining easily.
        """
    try:
        _need_restart = True
        res = proxy.run_one_command(cmd, stdin)
        _need_restart = False
        return res
    finally:
        if _need_restart:
            self._need_restart = True