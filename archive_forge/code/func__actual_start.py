import errno
import os
import pdb
import socket
import stat
import struct
import sys
import time
import traceback
import gflags as flags
def _actual_start():
    """Another layer in the starting stack."""
    tb = None
    try:
        raise ZeroDivisionError('')
    except ZeroDivisionError:
        tb = sys.exc_info()[2]
    assert tb
    prev_prev_frame = tb.tb_frame.f_back.f_back
    if not prev_prev_frame:
        return
    prev_prev_name = prev_prev_frame.f_globals.get('__name__', None)
    if prev_prev_name != '__main__' and (not prev_prev_name.endswith('.appcommands')):
        return
    del tb
    sys.exc_clear()
    try:
        really_start()
    except SystemExit as e:
        raise
    except Exception as e:
        for handler in EXCEPTION_HANDLERS:
            try:
                if handler.Wants(e):
                    handler.Handle(e)
            except:
                sys.stderr.write(traceback.format_exc())
                raise
        raise