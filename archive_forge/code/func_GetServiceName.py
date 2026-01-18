from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
import six.moves.urllib.parse
def GetServiceName(api_version=DEFAULT_API_VERSION):
    """Gets the service name based on the configured API endpoint."""
    endpoint = apis.GetEffectiveApiEndpoint('privateca', api_version)
    return six.moves.urllib.parse.urlparse(endpoint).hostname