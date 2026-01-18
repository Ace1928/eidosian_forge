from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.anthos import binary_operations
class PackOperation(binary_operations.BinaryBackedOperation):
    """PackOperation is a wrapper of the package-go-module binary."""

    def __init__(self, **kwargs):
        super(PackOperation, self).__init__(binary='package-go-module', **kwargs)

    def _ParseArgsForCommand(self, module_path, version, source, output, **kwargs):
        args = ['--module_path=' + module_path, '--version=' + version, '--source=' + source, '--output=' + output]
        return args