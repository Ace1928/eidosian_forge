import os
import sys
import time
import errno
import signal
import warnings
import subprocess
import traceback
def _kill_process_tree_with_psutil(process):
    try:
        descendants = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for descendant in descendants[::-1]:
        try:
            descendant.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        psutil.Process(process.pid).kill()
    except psutil.NoSuchProcess:
        pass
    process.join()