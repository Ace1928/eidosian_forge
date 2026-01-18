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
def _SkipReplicas(msgs, skip_replicas_arg, replica_info_list):
    """Skips each in skip_replicas_arg from the given ReplicaInfo list."""
    for replica_to_skip in skip_replicas_arg:
        index_to_delete = None
        replica_type = arg_utils.ChoiceToEnum(replica_to_skip['type'], msgs.ReplicaInfo.TypeValueValuesEnum)
        for index, replica in enumerate(replica_info_list):
            if replica.location == replica_to_skip['location'] and replica.type == replica_type:
                index_to_delete = index
                pass
        if index_to_delete is None:
            raise MissingReplicaError(replica_to_skip['location'], replica_type)
        replica_info_list.pop(index_to_delete)