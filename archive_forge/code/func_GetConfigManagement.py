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
def GetConfigManagement(membership):
    """Get ConfigManagement to check if multi-repo is enabled.

  Args:
    membership: The membership name or cluster name of the current cluster.

  Raises:
    Error: errors that happen when getting the object from the cluster.
  """
    config_management = None
    err = None
    timed_out = True
    with Timeout(_KUBECTL_TIMEOUT):
        config_management, err = RunKubectl(['get', 'configmanagements.configmanagement.gke.io/config-management', '-o', 'json'])
        timed_out = False
    if timed_out:
        if IsConnectGatewayContext():
            raise exceptions.ConfigSyncError('Timed out getting ConfigManagement object. ' + 'Make sure you have setup Connect Gateway for ' + membership + ' following the instruction from ' + 'https://cloud.google.com/anthos/multicluster-management/gateway/setup')
        else:
            raise exceptions.ConfigSyncError('Timed out getting ConfigManagement object from ' + membership)
    if err:
        raise exceptions.ConfigSyncError('Error getting ConfigManagement object from {}: {}\n'.format(membership, err))
    config_management_obj = json.loads(config_management)
    if 'enableMultiRepo' not in config_management_obj['spec'] or not config_management_obj['spec']['enableMultiRepo']:
        raise exceptions.ConfigSyncError('Legacy mode is used in {}. '.format(membership) + 'Please enable the multi-repo feature to use this command.')
    if 'status' not in config_management_obj:
        log.status.Print('The ConfigManagement object is not reconciled in {}. '.format(membership) + 'Please check if the Config Management is running on it.')
    errors = config_management_obj.get('status', {}).get('errors')
    if errors:
        log.status.Print('The ConfigManagement object contains errors in{}:\n{}'.format(membership, errors))