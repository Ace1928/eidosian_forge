import atexit
import os
import signal
import sys
import ovs.vlog
Like fatal_signal_remove_file_to_unlink(), but also unlinks 'file'.
    Returns 0 if successful, otherwise a positive errno value.