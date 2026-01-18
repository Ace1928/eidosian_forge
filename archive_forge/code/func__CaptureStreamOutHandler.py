from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import os
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kuberun import auth
from googlecloudsdk.command_lib.kuberun import flags
from googlecloudsdk.command_lib.kuberun import kuberuncli
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.console import console_io
def _CaptureStreamOutHandler(result_holder, **kwargs):
    """Captures streaming stdout from subprocess for processing in result handlers."""
    del kwargs

    def HandleStdOut(line):
        if line:
            line.rstrip()
            if not result_holder.stdout:
                result_holder.stdout = line
            else:
                result_holder.stdout += '\n' + line
    return HandleStdOut