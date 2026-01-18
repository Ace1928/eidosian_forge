from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import subprocess
import sys
from googlecloudsdk.command_lib.code import run_subprocess
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import times
import six
def _NamespaceExists(namespace, context_name=None):
    cmd = [_FindKubectl()]
    if context_name:
        cmd += ['--context', context_name]
    cmd += ['get', 'namespaces', '-o', 'name']
    namespaces = run_subprocess.GetOutputLines(cmd, timeout_sec=20, show_stderr=False)
    return 'namespace/' + namespace in namespaces