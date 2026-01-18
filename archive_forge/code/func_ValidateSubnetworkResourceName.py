from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
from typing import Any
from googlecloudsdk.calliope import exceptions
def ValidateSubnetworkResourceName(arg_name):
    """Validates the resource name of a compute subnetwork, must be in the form 'projects/{project_id}/regions/{region_id}/subnetworks/{subnetwork_id}'."""

    def Process(resource_name):
        pattern = re.compile('^projects/[^/]+/regions/[^/]+/subnetworks/[^/]+$')
        if not pattern.match(resource_name):
            raise exceptions.BadArgumentException(arg_name, 'The subnetwork resource name should be in the format projects/{project_id}/regions/{region_id}/subnetworks/{subnetwork_id}')
        return resource_name
    return Process