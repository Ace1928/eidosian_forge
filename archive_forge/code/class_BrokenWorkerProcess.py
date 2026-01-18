from __future__ import annotations
class BrokenWorkerProcess(Exception):
    """
    Raised by :func:`run_sync_in_process` if the worker process terminates abruptly or
    otherwise misbehaves.
    """