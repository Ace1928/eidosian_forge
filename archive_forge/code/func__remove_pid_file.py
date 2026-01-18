import atexit
import errno
import logging
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from paste.deploy import loadapp, loadserver
from paste.script.command import Command, BadCommand
def _remove_pid_file(written_pid, filename, verbosity):
    current_pid = os.getpid()
    if written_pid != current_pid:
        return
    if not os.path.exists(filename):
        return
    f = open(filename)
    content = f.read().strip()
    f.close()
    try:
        pid_in_file = int(content)
    except ValueError:
        pass
    else:
        if pid_in_file != current_pid:
            print('PID file %s contains %s, not expected PID %s' % (filename, pid_in_file, current_pid))
            return
    if verbosity > 0:
        print('Removing PID file %s' % filename)
    try:
        os.unlink(filename)
        return
    except OSError as e:
        print('Cannot remove PID file: %s' % e)
    try:
        f = open(filename, 'w')
        f.write('')
        f.close()
    except OSError as e:
        print('Stale PID left in file: %s (%e)' % (filename, e))
    else:
        print('Stale PID removed')