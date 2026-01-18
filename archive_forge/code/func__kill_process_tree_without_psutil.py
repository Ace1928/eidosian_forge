import os
import sys
import time
import errno
import signal
import warnings
import subprocess
import traceback
def _kill_process_tree_without_psutil(process):
    """Terminate a process and its descendants."""
    try:
        if sys.platform == 'win32':
            _windows_taskkill_process_tree(process.pid)
        else:
            _posix_recursive_kill(process.pid)
    except Exception:
        details = traceback.format_exc()
        warnings.warn(f'Failed to kill subprocesses on this platform. Please installpsutil: https://github.com/giampaolo/psutil\nDetails:\n{details}')
        process.kill()
    process.join()