from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instances import exceptions
from googlecloudsdk.command_lib.compute.instances import flags
def _original_scopes(self, instance_ref, client):
    """Return scopes instance is using."""
    instance = self._get_instance(instance_ref, client)
    if instance is None:
        return []
    orignal_service_accounts = instance.serviceAccounts
    result = []
    for accounts in orignal_service_accounts:
        result += accounts.scopes
    return result