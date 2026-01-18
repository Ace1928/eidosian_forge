from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import properties
def ValidatePlatformIsManaged(unused_ref, unused_args, req):
    """Validate the specified platform is managed.

  This method is referenced by the declarative iam commands which only work
  against the managed platform.

  Args:
    unused_ref: ref to the service.
    unused_args: Namespace, The args namespace.
    req: The request to be made.

  Returns:
    Unmodified request
  """
    if GetPlatform() != PLATFORM_MANAGED:
        raise calliope_exceptions.BadArgumentException('--platform', 'The platform [{platform}] is not supported by this operation. Specify `--platform {managed}` or run `gcloud config set run/platform {managed}`.'.format(platform=GetPlatform(), managed=PLATFORM_MANAGED))
    return req