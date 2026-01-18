from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def AwaitBackup(operation_ref, message):
    """Waits for backup long running operation to complete."""
    client = GetAdminClient()
    return _Await(client.projects_instances_clusters_backups, operation_ref, message)