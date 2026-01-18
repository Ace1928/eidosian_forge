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
def DeleteMinikube(cluster_name):
    """Delete a minikube cluster."""
    cmd = [_FindMinikube(), 'delete', '-p', cluster_name]
    print("Deleting development environment '%s' ..." % cluster_name)
    run_subprocess.Run(cmd, timeout_sec=150, show_output=False)
    print('Development environment stopped.')