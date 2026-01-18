from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import property_selector
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.backend_services import backend_services_utils
from googlecloudsdk.command_lib.compute.backend_services import flags
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import edit
import six
def GetExampleResource(self, client):
    uri_prefix = 'https://compute.googleapis.com/compute/v1/projects/my-project/'
    instance_groups_uri_prefix = 'https://compute.googleapis.com/compute/v1/projects/my-project/zones/'
    return client.messages.BackendService(backends=[client.messages.Backend(balancingMode=client.messages.Backend.BalancingModeValueValuesEnum.RATE, group=instance_groups_uri_prefix + 'us-central1-a/instanceGroups/group-1', maxRate=100), client.messages.Backend(balancingMode=client.messages.Backend.BalancingModeValueValuesEnum.RATE, group=instance_groups_uri_prefix + 'europe-west1-a/instanceGroups/group-2', maxRate=150)], customRequestHeaders=['X-Forwarded-Port:443'], customResponseHeaders=['X-Client-Geo-Location:US,Mountain View'], description='My backend service', healthChecks=[uri_prefix + 'global/httpHealthChecks/my-health-check-1', uri_prefix + 'global/httpHealthChecks/my-health-check-2'], name='backend-service', port=80, portName='http', protocol=client.messages.BackendService.ProtocolValueValuesEnum.HTTP, selfLink=uri_prefix + 'global/backendServices/backend-service', timeoutSec=30)