from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import threading
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import gce_cache
from googlecloudsdk.core.credentials import gce_read
from googlecloudsdk.core.util import retry
from six.moves import urllib
class _GCEMetadata(object):
    """Class for fetching GCE metadata.

  Attributes:
      connected: bool, True if the metadata server is available.
  """

    def __init__(self):
        self.connected = gce_cache.GetOnGCE()

    @_HandleMissingMetadataServer()
    def DefaultAccount(self):
        """Get the default service account for the host GCE instance.

    Fetches GOOGLE_GCE_METADATA_DEFAULT_ACCOUNT_URI and returns its contents.

    Raises:
      CannotConnectToMetadataServerException: If the metadata server
          cannot be reached.
      MetadataServerException: If there is a problem communicating with the
          metadata server.

    Returns:
      str, The email address for the default service account. None if not on a
          GCE VM, or if there are no service accounts associated with this VM.
    """
        if not self.connected:
            return None
        account = _ReadNoProxyWithCleanFailures(gce_read.GOOGLE_GCE_METADATA_DEFAULT_ACCOUNT_URI, http_errors_to_ignore=(404,))
        if account == CLOUDTOP_COMMON_SERVICE_ACCOUNT:
            return None
        return account

    @_HandleMissingMetadataServer()
    def Project(self):
        """Get the project that owns the current GCE instance.

    Fetches GOOGLE_GCE_METADATA_PROJECT_URI and returns its contents.

    Raises:
      CannotConnectToMetadataServerException: If the metadata server
          cannot be reached.
      MetadataServerException: If there is a problem communicating with the
          metadata server.

    Returns:
      str, The project ID for the current active project. None if no project is
          currently active.
    """
        if not self.connected:
            return None
        project = _ReadNoProxyWithCleanFailures(gce_read.GOOGLE_GCE_METADATA_PROJECT_URI)
        if project:
            return project
        return None

    @_HandleMissingMetadataServer(return_list=True)
    def Accounts(self):
        """Get the list of service accounts available from the metadata server.

    Returns:
      [str], The list of accounts. [] if not on a GCE VM.

    Raises:
      CannotConnectToMetadataServerException: If no metadata server is present.
      MetadataServerException: If there is a problem communicating with the
          metadata server.
    """
        if not self.connected:
            return []
        accounts_listing = _ReadNoProxyWithCleanFailures(gce_read.GOOGLE_GCE_METADATA_ACCOUNTS_URI + '/')
        accounts_lines = accounts_listing.split()
        accounts = []
        for account_line in accounts_lines:
            account = account_line.strip('/')
            if account == 'default' or account == CLOUDTOP_COMMON_SERVICE_ACCOUNT:
                continue
            accounts.append(account)
        return accounts

    @_HandleMissingMetadataServer()
    def Zone(self):
        """Get the name of the zone containing the current GCE instance.

    Fetches GOOGLE_GCE_METADATA_ZONE_URI, formats it, and returns its contents.

    Raises:
      CannotConnectToMetadataServerException: If the metadata server
          cannot be reached.
      MetadataServerException: If there is a problem communicating with the
          metadata server.

    Returns:
      str, The short name (e.g., us-central1-f) of the zone containing the
          current instance.
      None if not on a GCE VM.
    """
        if not self.connected:
            return None
        zone_path = _ReadNoProxyWithCleanFailures(gce_read.GOOGLE_GCE_METADATA_ZONE_URI)
        return zone_path.split('/')[-1]

    def Region(self):
        """Get the name of the region containing the current GCE instance.

    Fetches GOOGLE_GCE_METADATA_ZONE_URI, extracts the region associated
    with the zone, and returns it.  Extraction is based property that
    zone names have form <region>-<zone> (see https://cloud.google.com/
    compute/docs/zones) and an assumption that <zone> contains no hyphens.

    Raises:
      CannotConnectToMetadataServerException: If the metadata server
          cannot be reached.
      MetadataServerException: If there is a problem communicating with the
          metadata server.

    Returns:
      str, The short name (e.g., us-central1) of the region containing the
          current instance.
      None if not on a GCE VM.
    """
        if not self.connected:
            return None
        zone = self.Zone()
        return '-'.join(zone.split('-')[:-1]) if zone else None

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

    @_HandleMissingMetadataServer()
    def UniverseDomain(self):
        """Get the universe domain of the current GCE instance.

    If the GCE metadata server universe domain endpoint is not found, or the
    endpoint returns an empty string, return the default universe domain
    (googleapis.com); otherwise return the fetched universe domain value, or
    raise an exception if the request fails.

    Raises:
      CannotConnectToMetadataServerException: If the metadata server
          cannot be reached.
      MetadataServerException: If there is a problem communicating with the
          metadata server.

    Returns:
      str, The universe domain value from metadata server. None if not on GCE.
    """
        if not self.connected:
            return None
        universe_domain = _ReadNoProxyWithCleanFailures(gce_read.GOOGLE_GCE_METADATA_UNIVERSE_DOMAIN_URI, http_errors_to_ignore=(404,))
        if not universe_domain:
            return properties.VALUES.core.universe_domain.default
        return universe_domain