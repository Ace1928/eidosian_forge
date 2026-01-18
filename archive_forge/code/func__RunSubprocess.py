from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
import subprocess
import textwrap
from gae_ext_runtime import ext_runtime
from googlecloudsdk.api_lib.app import ext_runtime_adapter
from googlecloudsdk.api_lib.app.images import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
def _RunSubprocess(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    if p.wait() != 0:
        raise RubyConfigError('Unable to run script: [{0}]'.format(cmd))
    return p.stdout.read()