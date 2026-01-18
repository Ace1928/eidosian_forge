from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import os
from googlecloudsdk.command_lib.emulators import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
import six
def _BuildStartArgsForNativeExecutable(args):
    spanner_executable = os.path.join(util.GetCloudSDKRoot(), 'bin', SPANNER_EMULATOR_EXECUTABLE_DIR, SPANNER_EMULATOR_EXECUTABLE_FILE)
    if args.host_port.port is None:
        raise InvalidHostPortFormat('Invalid value for --host-port. Must be in the format host:port')
    return execution_utils.ArgsForExecutableTool(spanner_executable, '--hostname', args.host_port.host, '--grpc_port', args.host_port.port, '--http_port', six.text_type(args.rest_port), '--enable_fault_injection' if args.enable_fault_injection else '')