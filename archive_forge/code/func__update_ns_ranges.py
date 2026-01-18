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
def _update_ns_ranges(self, p, ranges):
    for ns_name, ns_opts in list(p.namespaces.items()):
        if ',' in ns_name or (ranges and '-' in ns_name):
            for subns in self._parse_ns_range(ns_name, ranges):
                p.namespaces[subns].update(ns_opts)
            p.namespaces.pop(ns_name)