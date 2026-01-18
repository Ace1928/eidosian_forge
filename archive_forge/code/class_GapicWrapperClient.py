from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import gapic_util
from googlecloudsdk.generated_clients.gapic_clients import spanner_v1
class GapicWrapperClient(object):
    """Spanner client."""
    types = spanner_v1.types

    def __init__(self, credentials, **kwargs):
        """
    Instantiates the GapicWrapperClient for spanner_v1.

    Args:
      credentials: google.auth.credentials.Credentials, the credentials to use.
      **kwargs: Additional kwargs to pass to gapic.MakeClient.

    Returns:
        GapicWrapperClient
    """
        self.credentials = credentials
        self.spanner = gapic_util.MakeClient(spanner_v1.services.spanner.client.SpannerClient, credentials, **kwargs)