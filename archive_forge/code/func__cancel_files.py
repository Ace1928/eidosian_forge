import atexit
import os
import signal
import sys
import ovs.vlog
def _cancel_files():
    global _added_hook
    global _files
    _added_hook = False
    _files = {}