from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kuberun import flags
from googlecloudsdk.command_lib.kuberun import kuberun_command
def _TypeFlag():
    return flags.StringFlag('--type', help='Type of Component to create. See `kuberun devkits describe` for available Component Types.', required=True)