import os
import shutil
import sys
from ._process_common import getoutputerror, get_output_error_code, process_handler
def abbrev_cwd():
    """ Return abbreviated version of cwd, e.g. d:mydir """
    cwd = os.getcwd().replace('\\', '/')
    drivepart = ''
    tail = cwd
    if sys.platform == 'win32':
        if len(cwd) < 4:
            return cwd
        drivepart, tail = os.path.splitdrive(cwd)
    parts = tail.split('/')
    if len(parts) > 2:
        tail = '/'.join(parts[-2:])
    return drivepart + (cwd == '/' and '/' or tail)