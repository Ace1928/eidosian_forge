from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import threading
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import gce_cache
from googlecloudsdk.core.credentials import gce_read
from googlecloudsdk.core.util import retry
from six.moves import urllib
@_HandleMissingMetadataServer()
def GetIdToken(self, audience, token_format='standard', include_license=False):
    """Get a valid identity token on the host GCE instance.

    Fetches GOOGLE_GCE_METADATA_ID_TOKEN_URI and returns its contents.

    Args:
      audience: str, target audience for ID token.
      token_format: str, Specifies whether or not the project and instance
        details are included in the identity token. Choices are "standard",
        "full".
      include_license: bool, Specifies whether or not license codes for images
        associated with GCE instance are included in their identity tokens

    Raises:
      CannotConnectToMetadataServerException: If the metadata server
          cannot be reached.
      MetadataServerException: If there is a problem communicating with the
          metadata server.
      MissingAudienceForIdTokenError: If audience is missing.

    Returns:
      str, The id token or None if not on a CE VM, or if there are no
      service accounts associated with this VM.
    """
    if not self.connected:
        return None
    if not audience:
        raise MissingAudienceForIdTokenError()
    include_license = 'TRUE' if include_license else 'FALSE'
    return _ReadNoProxyWithCleanFailures(gce_read.GOOGLE_GCE_METADATA_ID_TOKEN_URI.format(audience=audience, format=token_format, licenses=include_license), http_errors_to_ignore=(404,))