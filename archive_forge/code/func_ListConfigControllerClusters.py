from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import fnmatch
import io
import json
import os
import re
import signal
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.command_lib.anthos.config.sync.common import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
def ListConfigControllerClusters(project):
    """Runs a gcloud command to list the clusters that host Config Controller.

  Currently the Config Controller only works in select regions.
  Refer to the Config Controller doc:
  https://cloud.google.com/anthos-config-management/docs/how-to/config-controller-setup

  Args:
    project: project that the Config Controller is in.

  Returns:
    The list of (cluster, region) for Config Controllers.

  Raises:
    Error: The error occured when it failed to list clusters.
  """
    args = ['container', 'clusters', 'list', '--project', project, '--filter', 'name:krmapihost', '--format', 'json(name,location)']
    output, err = _RunGcloud(args)
    if err:
        raise exceptions.ConfigSyncError('Error listing clusters: {}'.format(err))
    output_json = json.loads(output)
    clusters = [(c['name'], c['location']) for c in output_json]
    return clusters