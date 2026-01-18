from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.command_lib.emulators import util
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
def ArgsForFirestoreEmulator(emulator_args):
    """Constructs an argument list for calling the Firestore emulator.

  Args:
    emulator_args: args for the emulator.

  Returns:
    An argument list to execute the Firestore emulator.
  """
    current_os = platforms.OperatingSystem.Current()
    if current_os is platforms.OperatingSystem.WINDOWS:
        cmd = 'cloud_firestore_emulator.cmd'
        exe = os.path.join(util.GetEmulatorRoot(CLOUD_FIRESTORE), cmd)
        return execution_utils.ArgsForCMDTool(exe, *emulator_args)
    else:
        cmd = 'cloud_firestore_emulator'
        exe = os.path.join(util.GetEmulatorRoot(CLOUD_FIRESTORE), cmd)
        return execution_utils.ArgsForExecutableTool(exe, *emulator_args)