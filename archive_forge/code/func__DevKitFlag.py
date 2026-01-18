from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kuberun import flags
from googlecloudsdk.command_lib.kuberun import kuberun_command
def _DevKitFlag():
    return flags.StringFlag('--devkit', help='Name of the Development Kit to use for this Component.', required=True)