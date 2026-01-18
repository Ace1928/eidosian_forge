from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instances import exceptions
from googlecloudsdk.command_lib.compute.instances import flags
def _email(self, args, instance_ref, client):
    """Return email to set as service account for the instance."""
    if args.no_service_account:
        return None
    if args.service_account:
        return args.service_account
    return self._original_email(instance_ref, client)