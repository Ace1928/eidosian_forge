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
def _waitexec(self, argv, path=sys.executable, env=None, on_spawn=None, on_signalled=None, on_failure=None):
    argstr = self.prepare_argv(argv, path)
    maybe_call(on_spawn, self, argstr=' '.join(argstr), env=env)
    pipe = Popen(argstr, env=env)
    return self.handle_process_exit(pipe.wait(), on_signalled=on_signalled, on_failure=on_failure)