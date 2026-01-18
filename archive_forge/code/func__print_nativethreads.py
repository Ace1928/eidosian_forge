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
def _print_nativethreads():
    for threadId, stack in sys._current_frames().items():
        print(threadId)
        traceback.print_stack(stack)
        print()