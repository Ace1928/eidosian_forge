from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.command_lib.emulators import util
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
def StartFirestoreEmulator(args, log_file=None):
    """Starts the firestore emulator with the given arguments.

  Args:
    args: Arguments passed to the start command.
    log_file: optional file argument to reroute process's output.

  Returns:
    process, The handle of the child process running the datastore emulator.
  """
    start_args = ['start']
    start_args.append('--host={0}'.format(args.host_port.host))
    start_args.append('--port={0}'.format(args.host_port.port))
    if args.rules:
        start_args.append('--rules={0}'.format(args.rules))
    if args.use_firestore_in_datastore_mode:
        start_args.append('--database-mode=datastore-mode')
    else:
        start_args.append('--database-mode={0}'.format(args.database_mode))
    exec_args = ArgsForFirestoreEmulator(start_args)
    log.status.Print('Executing: {0}'.format(' '.join(exec_args)))
    return util.Exec(exec_args, log_file=log_file)