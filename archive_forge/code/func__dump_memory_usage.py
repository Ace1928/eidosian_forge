from both of those two places to another location.
import errno
import logging
import os
import sys
import time
from io import StringIO
import breezy
from .lazy_import import lazy_import
from breezy import (
from . import errors
def _dump_memory_usage(err_file):
    import tempfile
    try:
        try:
            fd, name = tempfile.mkstemp(prefix='brz_memdump', suffix='.json')
            dump_file = os.fdopen(fd, 'w')
            from meliae import scanner
            scanner.dump_gc_objects(dump_file)
            err_file.write('Memory dumped to %s\n' % name)
        except ImportError:
            err_file.write('Dumping memory requires meliae module.\n')
            log_exception_quietly()
        except BaseException:
            err_file.write('Exception while dumping memory.\n')
            log_exception_quietly()
    finally:
        if dump_file is not None:
            dump_file.close()
        elif fd is not None:
            os.close(fd)