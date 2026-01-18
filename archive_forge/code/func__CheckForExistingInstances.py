from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import daisy_utils
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute.images import os_choices
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.compute.sole_tenancy import flags as sole_tenancy_flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _CheckForExistingInstances(self, instance_name, client):
    """Check that the destination instances do not already exist."""
    zone = properties.VALUES.compute.zone.GetOrFail()
    request = (client.apitools_client.instances, 'Get', client.messages.ComputeInstancesGetRequest(instance=instance_name, project=properties.VALUES.core.project.GetOrFail(), zone=zone))
    errors = []
    instances = client.MakeRequests([request], errors_to_collect=errors)
    if not errors and instances:
        message = 'The VM instance [{instance_name}] already exists in zone [{zone}].'.format(instance_name=instance_name, zone=zone)
        raise exceptions.InvalidArgumentException('INSTANCE_NAME', message)