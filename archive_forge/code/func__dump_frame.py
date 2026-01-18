import errno
import gc
import logging
import os
import pprint
import sys
import tempfile
import traceback
import eventlet.backdoor
import greenlet
import yappi
from eventlet.green import socket
from oslo_service._i18n import _
from oslo_service import _options
def _dump_frame(f, frame_chapter):
    co = f.f_code
    print(' %s Frame: %s' % (frame_chapter, co.co_name))
    print('     File: %s' % co.co_filename)
    print('     Captured at line number: %s' % f.f_lineno)
    co_locals = set(co.co_varnames)
    if len(co_locals):
        not_set = co_locals.copy()
        set_locals = {}
        for var_name in f.f_locals.keys():
            if var_name in co_locals:
                set_locals[var_name] = f.f_locals[var_name]
                not_set.discard(var_name)
        if set_locals:
            print('     %s set local variables:' % len(set_locals))
            for var_name in sorted(set_locals.keys()):
                print('       %s => %r' % (var_name, f.f_locals[var_name]))
        else:
            print('     0 set local variables.')
        if not_set:
            print('     %s not set local variables:' % len(not_set))
            for var_name in sorted(not_set):
                print('       %s' % var_name)
        else:
            print('     0 not set local variables.')
    else:
        print('     0 Local variables.')