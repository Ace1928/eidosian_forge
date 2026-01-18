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
def _GetBaseUrlFromApi(api_name, api_version):
    """Returns base url for given api."""
    if GetApiDef(api_name, api_version).apitools:
        client_class = _GetClientClass(api_name, api_version)
    else:
        client_class = _GetGapicClientClass(api_name, api_version)
    if hasattr(client_class, 'BASE_URL'):
        client_base_url = client_class.BASE_URL
    else:
        try:
            client_base_url = _GetResourceModule(api_name, api_version).BASE_URL
        except AttributeError:
            client_base_url = 'https://{}.googleapis.com/{}'.format(api_name, api_version)
    return UniversifyAddress(client_base_url)