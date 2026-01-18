from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import datetime
import os.path
import signal
import subprocess
import sys
import threading
from googlecloudsdk.command_lib.code import json_stream
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files as file_utils
import six
@contextlib.contextmanager
def Skaffold(skaffold_config, context_name=None, namespace=None, env_vars=None, debug=False, events_port=None):
    """Run skaffold and catch keyboard interrupts to kill the process.

  Args:
    skaffold_config: Path to skaffold configuration yaml file.
    context_name: Kubernetes context name.
    namespace: Kubernetes namespace name.
    env_vars: Additional environment variables with which to run skaffold.
    debug: If true, turn on debugging output.
    events_port: If set, turn on the events api and expose it on this port.

  Yields:
    The skaffold process.
  """
    cmd = [_FindSkaffold(), 'dev', '-f', skaffold_config, '--port-forward']
    if context_name:
        cmd += ['--kube-context=%s' % context_name]
    if namespace:
        cmd += ['--namespace=%s' % namespace]
    if debug:
        cmd += ['-vdebug']
    if events_port:
        cmd += ['--rpc-http-port=%s' % events_port]
    with _SigInterruptedHandler(_KeyboardInterruptHandler):
        env = os.environ.copy()
        if env_vars:
            env.update(((six.ensure_str(name), six.ensure_str(value)) for name, value in env_vars.items()))
        if config.Paths().sdk_root:
            env['PATH'] = six.ensure_str(env['PATH'] + os.pathsep + config.Paths().sdk_root)
        try:
            p = subprocess.Popen(cmd, env=env)
            yield p
        except KeyboardInterrupt:
            p.terminate()
            p.wait()
        sys.stdout.flush()
        sys.stderr.flush()