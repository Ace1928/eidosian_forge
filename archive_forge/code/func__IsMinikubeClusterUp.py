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
def _IsMinikubeClusterUp(cluster_name):
    """Checks if a minikube cluster is running."""
    cmd = [_FindMinikube(), 'status', '-p', cluster_name, '-o', 'json']
    try:
        status = run_subprocess.GetOutputJson(cmd, timeout_sec=20, show_stderr=False)
        return 'Host' in status and status['Host'].strip() == 'Running'
    except (ValueError, subprocess.CalledProcessError):
        return False