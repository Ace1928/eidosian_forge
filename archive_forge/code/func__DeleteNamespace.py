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
def _DeleteNamespace(namespace, context_name=None):
    cmd = [_FindKubectl()]
    if context_name:
        cmd += ['--context', context_name]
    cmd += ['delete', 'namespace', namespace]
    run_subprocess.Run(cmd, timeout_sec=20, show_output=False)