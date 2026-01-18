from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
import os
import subprocess
import sys
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.credentials import store
from googlecloudsdk.core.docker import client_lib
from googlecloudsdk.core.docker import constants
from googlecloudsdk.core.util import files
import six
def _SurfaceUnexpectedInfo(stdoutdata, stderrdata):
    """Reads docker's output and surfaces unexpected lines.

  Docker's CLI has a certain amount of chattiness, even on successes.

  Args:
    stdoutdata: The raw data output from the pipe given to Popen as stdout.
    stderrdata: The raw data output from the pipe given to Popen as stderr.
  """
    stdout = [s.strip() for s in stdoutdata.splitlines()]
    stderr = [s.strip() for s in stderrdata.splitlines()]
    for line in stdout:
        if line != 'Login Succeeded' and 'login credentials saved in' not in line:
            line = '%s%s' % (line, os.linesep)
            log.out.Print(line)
    for line in stderr:
        if not _IsExpectedErrorLine(line):
            line = '%s%s' % (line, os.linesep)
            log.status.Print(line)