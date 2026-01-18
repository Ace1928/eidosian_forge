from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis_util
from googlecloudsdk.api_lib.util import resource as resource_util
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.generated_clients.apis import apis_map
import six
from six.moves.urllib.parse import urljoin
from six.moves.urllib.parse import urlparse
def _BuildEndpointOverride(endpoint_override, base_url):
    """Constructs a normalized endpoint URI depending on the client base_url."""
    url_base = urlparse(base_url)
    url_endpoint_override = urlparse(endpoint_override)
    if url_base.path == '/' or url_endpoint_override.path != '/':
        return endpoint_override
    return urljoin('{}://{}'.format(url_endpoint_override.scheme, url_endpoint_override.netloc), url_base.path)