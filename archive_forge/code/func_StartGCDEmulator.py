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
def StartGCDEmulator(args, log_file=None):
    """Starts the datastore emulator with the given arguments.

  Args:
    args: Arguments passed to the start command.
    log_file: optional file argument to reroute process's output.

  Returns:
    process, The handle of the child process running the datastore emulator.
  """
    gcd_start_args = ['start']
    gcd_start_args.append('--host={0}'.format(args.host_port.host))
    gcd_start_args.append('--port={0}'.format(args.host_port.port))
    gcd_start_args.append('--store_on_disk={0}'.format(args.store_on_disk))
    gcd_start_args.append('--allow_remote_shutdown')
    if args.use_firestore_in_datastore_mode:
        gcd_start_args.append('--firestore_in_datastore_mode')
    else:
        gcd_start_args.append('--consistency={0}'.format(args.consistency))
    gcd_start_args.append(args.data_dir)
    exec_args = ArgsForGCDEmulator(gcd_start_args)
    log.status.Print('Executing: {0}'.format(' '.join(exec_args)))
    return util.Exec(exec_args, log_file=log_file)