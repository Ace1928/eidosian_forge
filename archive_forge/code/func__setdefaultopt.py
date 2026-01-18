import errno
import os
import shlex
import signal
import sys
from collections import OrderedDict, UserList, defaultdict
from functools import partial
from subprocess import Popen
from time import sleep
from kombu.utils.encoding import from_utf8
from kombu.utils.objects import cached_property
from celery.platforms import IS_WINDOWS, Pidfile, signal_name
from celery.utils.nodenames import gethostname, host_format, node_format, nodesplit
from celery.utils.saferepr import saferepr
def _setdefaultopt(self, d, alt, value):
    for opt in alt[1:]:
        try:
            return d[opt]
        except KeyError:
            pass
    value = d.setdefault(alt[0], os.path.normpath(value))
    dir_path = os.path.dirname(value)
    if dir_path and (not os.path.exists(dir_path)):
        os.makedirs(dir_path)
    return value