from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def AwaitInstance(operation_ref, message):
    """Waits for instance long running operation to complete."""
    client = GetAdminClient()
    return _Await(client.projects_instances, operation_ref, message)