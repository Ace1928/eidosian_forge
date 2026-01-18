from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kuberun import flags
from googlecloudsdk.command_lib.kuberun import kuberun_command
def _DirectoryFlag():
    return flags.StringFlag('--directory', help='Base directory path relative to current working directory where Component will be created.  This path must be within the application git repository.', required=False)