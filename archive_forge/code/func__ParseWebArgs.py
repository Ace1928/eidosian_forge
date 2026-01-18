from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import os
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import exceptions as c_except
def _ParseWebArgs(self, open_flag=False, port=None, **kwargs):
    """Parse args for the web command."""
    del kwargs
    exec_args = ['web']
    if open_flag:
        exec_args.append('--open')
    if port:
        exec_args.extend(['--port', port])
    return exec_args