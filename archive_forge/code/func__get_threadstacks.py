import collections
import gc
import io
import ipaddress
import json
import platform
import socket
import sys
import traceback
from debtcollector import removals
import jinja2
from oslo_utils import reflection
from oslo_utils import strutils
from oslo_utils import timeutils
import stevedore
import webob.dec
import webob.exc
import webob.response
from oslo_middleware import base
from oslo_middleware.healthcheck import opts
@staticmethod
def _get_threadstacks():
    threadstacks = []
    try:
        active_frames = sys._current_frames()
    except AttributeError:
        pass
    else:
        buf = io.StringIO()
        for stack in active_frames.values():
            traceback.print_stack(stack, file=buf)
            threadstacks.append(buf.getvalue())
            buf.seek(0)
            buf.truncate()
    return threadstacks