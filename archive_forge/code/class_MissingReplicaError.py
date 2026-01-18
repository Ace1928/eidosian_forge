from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
class MissingReplicaError(core_exceptions.Error):
    """Indicates that the replica is missing in the source config."""

    def __init__(self, replica_location, replica_type):
        super(MissingReplicaError, self).__init__("The replica {0} of type {1} is not in the source config's replicas".format(replica_location, replica_type))