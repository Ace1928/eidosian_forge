from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import tempfile
from googlecloudsdk.command_lib.emulators import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import platforms
def ArgsForGCDEmulator(emulator_args):
    """Constructs an argument list for calling the GCD emulator.

  Args:
    emulator_args: args for the emulator.

  Returns:
    An argument list to execute the GCD emulator.
  """
    current_os = platforms.OperatingSystem.Current()
    if current_os is platforms.OperatingSystem.WINDOWS:
        cmd = 'cloud_datastore_emulator.cmd'
        gcd_executable = os.path.join(util.GetEmulatorRoot(CLOUD_DATASTORE), cmd)
        return execution_utils.ArgsForCMDTool(gcd_executable, *emulator_args)
    else:
        cmd = 'cloud_datastore_emulator'
        gcd_executable = os.path.join(util.GetEmulatorRoot(CLOUD_DATASTORE), cmd)
        return execution_utils.ArgsForExecutableTool(gcd_executable, *emulator_args)