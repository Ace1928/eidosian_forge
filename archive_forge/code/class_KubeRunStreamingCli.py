from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import os
from googlecloudsdk.command_lib.kuberun import messages
from googlecloudsdk.command_lib.util.anthos import binary_operations
class KubeRunStreamingCli(binary_operations.StreamingBinaryBackedOperation):
    """Binary operation wrapper for kuberun commands that require streaming output."""

    def __init__(self, **kwargs):
        custom_errors = {'MISSING_EXEC': messages.MISSING_BINARY.format(binary='kuberun')}
        super(KubeRunStreamingCli, self).__init__(binary='kuberun', check_hidden=True, custom_errors=custom_errors, **kwargs)

    def _ParseArgsForCommand(self, command, **kwargs):
        return command