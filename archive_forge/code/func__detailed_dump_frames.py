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
def _detailed_dump_frames(f, thread_index):
    i = 0
    while f is not None:
        _dump_frame(f, '%s.%s' % (thread_index, i + 1))
        f = f.f_back
        i += 1