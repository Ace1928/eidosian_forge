from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core.util import text
def UpdateFieldRequestTtls(ref, args, request):
    """Update field request for TTL.

  Args:
    ref: The field resource reference(unused).
    args: The parsed arg namespace.
    request: The ttl field request.
  Raises:
    InvalidArgumentException: If the provided indexes are incorrectly specified.
  Returns:
    UpdateFieldRequest
  """
    messages = GetMessagesModule()
    request.updateMask = 'ttlConfig'
    ttl_config = None
    if args.enable_ttl:
        ttl_config = messages.GoogleFirestoreAdminV1TtlConfig()
    request.googleFirestoreAdminV1Field = messages.GoogleFirestoreAdminV1Field(name=ref.RelativeName(), ttlConfig=ttl_config)
    return request