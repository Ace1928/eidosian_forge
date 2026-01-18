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
def _MtlsEnabled(api_name, api_version):
    """Checks if the API of the given version should use mTLS.

  If context_aware/always_use_mtls_endpoint is True, then mTLS will always be
  used.

  If context_aware/use_client_certificate is True, then mTLS will be used only
  if the API version is in the mTLS allowlist.

  gcloud maintains a client-side allowlist for the mTLS feature
  (go/gcloud-rollout-mtls).

  Args:
    api_name: str, The API name.
    api_version: str, The version of the API.

  Returns:
    True if the given service and version is in the mTLS allowlist.
  """
    if properties.VALUES.context_aware.always_use_mtls_endpoint.GetBool():
        return True
    if not properties.VALUES.context_aware.use_client_certificate.GetBool():
        return False
    api_def = GetApiDef(api_name, api_version)
    return api_def.enable_mtls