from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.pubsub import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
from six.moves.urllib.parse import urlparse
def OverrideEndpointWithRegion(request):
    """Sets the pubsublite endpoint override to include the region."""
    resource, _ = GetResourceInfo(request)
    region = DeriveRegionFromLocation(DeriveLocationFromResource(resource))
    endpoint = apis.GetEffectiveApiEndpoint(PUBSUBLITE_API_NAME, PUBSUBLITE_API_VERSION)
    endpoint = RemoveRegionFromEndpoint(endpoint)
    regional_endpoint = CreateRegionalEndpoint(region, endpoint)
    properties.VALUES.api_endpoint_overrides.pubsublite.Set(regional_endpoint)