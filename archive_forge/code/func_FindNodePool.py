from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import time
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import http_wrapper
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.container import constants as gke_constants
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import times
import six
from six.moves import range  # pylint: disable=redefined-builtin
import six.moves.http_client
from cmd argument to set a surge upgrade strategy.
def FindNodePool(self, cluster, pool_name=None):
    """Find the node pool with the given name in the cluster."""
    msg = ''
    if pool_name:
        for np in cluster.nodePools:
            if np.name == pool_name:
                return np
        msg = NO_SUCH_NODE_POOL_ERROR_MSG.format(cluster=cluster.name, name=pool_name) + os.linesep
    elif len(cluster.nodePools) == 1:
        return cluster.nodePools[0]
    msg += NO_NODE_POOL_SELECTED_ERROR_MSG + os.linesep.join([np.name for np in cluster.nodePools])
    raise util.Error(msg)