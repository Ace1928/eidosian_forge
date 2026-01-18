from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
import re
import enum
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def _UpdateShieldedInstanceConfig(holder, client, operation_poller, instance_ref, args):
    """Update the Shielded Instance Config."""
    if args.shielded_vm_secure_boot is None and args.shielded_vm_vtpm is None and (args.shielded_vm_integrity_monitoring is None):
        return None
    shielded_config_msg = client.messages.ShieldedInstanceConfig(enableSecureBoot=args.shielded_vm_secure_boot, enableVtpm=args.shielded_vm_vtpm, enableIntegrityMonitoring=args.shielded_vm_integrity_monitoring)
    request = client.messages.ComputeInstancesUpdateShieldedInstanceConfigRequest(instance=instance_ref.Name(), project=instance_ref.project, shieldedInstanceConfig=shielded_config_msg, zone=instance_ref.zone)
    operation = client.apitools_client.instances.UpdateShieldedInstanceConfig(request)
    operation_ref = holder.resources.Parse(operation.selfLink, collection='compute.zoneOperations')
    return waiter.WaitFor(operation_poller, operation_ref, 'Setting shieldedInstanceConfig of instance [{0}]'.format(instance_ref.Name()))