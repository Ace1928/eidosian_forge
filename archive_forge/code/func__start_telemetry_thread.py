from queue import Queue
from threading import Lock, Thread
from typing import Dict, Optional, Union
from urllib.parse import quote
from .. import constants, logging
from . import build_hf_headers, get_session, hf_raise_for_status
def _start_telemetry_thread():
    """Start a daemon thread to consume tasks from the telemetry queue.

    If the thread is interrupted, start a new one.
    """
    with _TELEMETRY_THREAD_LOCK:
        global _TELEMETRY_THREAD
        if _TELEMETRY_THREAD is None or not _TELEMETRY_THREAD.is_alive():
            _TELEMETRY_THREAD = Thread(target=_telemetry_worker, daemon=True)
            _TELEMETRY_THREAD.start()