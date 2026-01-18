from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.iam import iam_util
def _MakeReplicationMessage(messages, policy, locations, keys):
    """Create a replication message from its components."""
    if not policy:
        return None
    replication = messages.Replication(automatic=messages.Automatic())
    if policy == 'automatic' and keys:
        replication = messages.Replication(automatic=messages.Automatic(customerManagedEncryption=messages.CustomerManagedEncryption(kmsKeyName=keys[0])))
    if policy == 'user-managed':
        replicas = []
        for i, location in enumerate(locations):
            if i < len(keys):
                replicas.append(messages.Replica(location=location, customerManagedEncryption=messages.CustomerManagedEncryption(kmsKeyName=keys[i])))
            else:
                replicas.append(messages.Replica(location=locations[i]))
        replication = messages.Replication(userManaged=messages.UserManaged(replicas=replicas))
    return replication