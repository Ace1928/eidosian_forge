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
def AddressOverride(address):
    try:
        endpoint_override = properties.VALUES.api_endpoint_overrides.Property(api_name).Get()
    except properties.NoSuchPropertyError:
        endpoint_override = None
    if endpoint_override:
        address = urlparse(endpoint_override).netloc
    if address_override_func:
        address = address_override_func(address)
    if endpoint_override is not None:
        return address
    return UniversifyAddress(address)