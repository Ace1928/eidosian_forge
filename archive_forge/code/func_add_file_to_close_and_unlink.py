import atexit
import os
import signal
import sys
import ovs.vlog
def add_file_to_close_and_unlink(file, fd=None):
    """Registers 'file' to be unlinked when the program terminates via
    sys.exit() or a fatal signal and the 'fd' to be closed. On Windows a file
    cannot be removed while it is open for writing."""
    global _added_hook
    if not _added_hook:
        _added_hook = True
        add_hook(_unlink_files, _cancel_files, True)
    _files[file] = fd