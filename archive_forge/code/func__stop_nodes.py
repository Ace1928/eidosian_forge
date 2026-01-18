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
def _stop_nodes(self, retry=None, on_down=None, sig=signal.SIGTERM):
    on_down = on_down if on_down is not None else self.on_node_down
    nodes = list(self.getpids(on_down=on_down))
    if nodes:
        for node in self.shutdown_nodes(nodes, sig=sig, retry=retry):
            maybe_call(on_down, node)