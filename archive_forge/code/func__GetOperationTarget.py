from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import util as gke_util
from googlecloudsdk.api_lib.container.gkemulticloud import operations as op_api_util
from googlecloudsdk.api_lib.container.gkemulticloud import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.run import pretty_print
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def _GetOperationTarget(op):
    op_cluster = ''
    if op.metadata is not None:
        metadata = encoding.MessageToPyValue(op.metadata)
        if 'target' in metadata:
            op_cluster = metadata['target']
    return resources.REGISTRY.ParseRelativeName(op_cluster, collection='gkemulticloud.projects.locations.attachedClusters')