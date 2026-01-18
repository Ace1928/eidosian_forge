import logging
import os
import sys
import threading
import importlib
import ray
from ray.util.annotations import DeveloperAPI
def _ensure_debugger_port_open_thread_safe():
    """
    This is a thread safe method that ensure that the debugger port
    is open, and if not, open it.
    """
    with debugger_port_lock:
        debugpy = _try_import_debugpy()
        if not debugpy:
            return
        debugger_port = ray._private.worker.global_worker.debugger_port
        if not debugger_port:
            host, port = debugpy.listen((ray._private.worker.global_worker.node_ip_address, 0))
            ray._private.worker.global_worker.set_debugger_port(port)
            log.info(f'Ray debugger is listening on {host}:{port}')
        else:
            log.info(f'Ray debugger is already open on {debugger_port}')