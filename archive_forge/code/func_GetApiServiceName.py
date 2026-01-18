from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
import six.moves.urllib.parse
def GetApiServiceName(api_version):
    """Gets the service name based on the configured API endpoint."""
    endpoint = apis.GetEffectiveApiEndpoint(API_NAME, api_version)
    return six.moves.urllib.parse.urlparse(endpoint).hostname