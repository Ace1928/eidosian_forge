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
def PrepareGCDDataDir(args):
    """Prepares the given directory using gcd create.

  Raises:
    UnableToPrepareDataDir: If the gcd create execution fails.

  Args:
    args: The arguments passed to the command.
  """
    data_dir = args.data_dir
    if os.path.isdir(data_dir) and os.listdir(data_dir):
        log.warning('Reusing existing data in [{0}].'.format(data_dir))
        return
    gcd_create_args = ['create']
    project = properties.VALUES.core.project.Get(required=True)
    gcd_create_args.append('--project_id={0}'.format(project))
    gcd_create_args.append(data_dir)
    exec_args = ArgsForGCDEmulator(gcd_create_args)
    log.status.Print('Executing: {0}'.format(' '.join(exec_args)))
    with util.Exec(exec_args) as process:
        util.PrefixOutput(process, DATASTORE)
        failed = process.poll()
        if failed:
            raise UnableToPrepareDataDir()