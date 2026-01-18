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
def WriteGCDEnvYaml(args):
    """Writes the env.yaml file for the datastore emulator with provided args.

  Args:
    args: Arguments passed to the start command.
  """
    host_port = '{0}:{1}'.format(args.host_port.host, args.host_port.port)
    project_id = properties.VALUES.core.project.Get(required=True)
    env = {'DATASTORE_HOST': 'http://{0}'.format(host_port), 'DATASTORE_EMULATOR_HOST': host_port, 'DATASTORE_EMULATOR_HOST_PATH': '{0}/datastore'.format(host_port), 'DATASTORE_DATASET': project_id, 'DATASTORE_PROJECT_ID': project_id}
    util.WriteEnvYaml(env, args.data_dir)