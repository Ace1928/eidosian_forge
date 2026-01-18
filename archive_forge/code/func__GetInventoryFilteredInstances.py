from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
import os
import zlib
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_filter
from googlecloudsdk.core.resource import resource_projector
def _GetInventoryFilteredInstances(self, instances, responses, query):
    filtered_instances = []
    for instance, response in zip(instances, responses):
        if instance is not None and response is not None:
            guest_attributes = response.queryValue.items
            formatted_guest_attributes_json = self._GetFormattedGuestAttributes(guest_attributes)
            if query.Evaluate(formatted_guest_attributes_json):
                filtered_instances.append(instance)
    return filtered_instances