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
def _StartVm(holder, client, instance_ref):
    """Start the Virtual Machine."""
    operation = client.apitools_client.instances.Start(client.messages.ComputeInstancesStartRequest(**instance_ref.AsDict()))
    operation_ref = holder.resources.Parse(operation.selfLink, collection='compute.zoneOperations')
    operation_poller = poller.Poller(client.apitools_client.instances)
    return waiter.WaitFor(operation_poller, operation_ref, 'Starting instance [{0}]'.format(instance_ref.Name()))