from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import gapic_util
from googlecloudsdk.generated_clients.gapic_clients import spanner_v1

    Instantiates the GapicWrapperClient for spanner_v1.

    Args:
      credentials: google.auth.credentials.Credentials, the credentials to use.
      **kwargs: Additional kwargs to pass to gapic.MakeClient.

    Returns:
        GapicWrapperClient
    