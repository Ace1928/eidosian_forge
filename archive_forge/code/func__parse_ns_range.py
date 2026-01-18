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
def _parse_ns_range(self, ns, ranges=False):
    ret = []
    for space in ',' in ns and ns.split(',') or [ns]:
        if ranges and '-' in space:
            start, stop = space.split('-')
            ret.extend((str(n) for n in range(int(start), int(stop) + 1)))
        else:
            ret.append(space)
    return ret