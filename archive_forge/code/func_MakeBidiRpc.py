from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from google.auth import external_account as google_auth_external_account
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import requests
from googlecloudsdk.core.credentials import creds
from googlecloudsdk.core.credentials import store
def MakeBidiRpc(client, start_rpc, initial_request=None):
    """Initializes a BidiRpc instances.

  Args:
      client: GAPIC Wrapper client to use.
      start_rpc (grpc.StreamStreamMultiCallable): The gRPC method used to
          start the RPC.
      initial_request: The initial request to
          yield. This is useful if an initial request is needed to start the
          stream.
  Returns:
    A bidiRPC instance.
  """
    from googlecloudsdk.core import gapic_util_internal
    return gapic_util_internal.BidiRpc(client, start_rpc, initial_request=initial_request)