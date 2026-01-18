from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
from typing import Any
from googlecloudsdk.calliope import exceptions
def ValidateNetworkResourceName(arg_name):
    """Validates the resource name of a compute network, must be in the form 'projects/{project_id}/global/networks/{network_id}'."""

    def Process(resource_name):
        pattern = re.compile('^projects/[^/]+/global/networks/[^/]+$')
        if not pattern.match(resource_name):
            raise exceptions.BadArgumentException(arg_name, 'The network resource name should be in the format projects/<project_id>/global/networks/<network_id>')
        return resource_name
    return Process