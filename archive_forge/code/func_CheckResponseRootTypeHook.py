from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.privateca import base
from googlecloudsdk.api_lib.privateca import request_utils
from googlecloudsdk.command_lib.privateca import resource_args
from googlecloudsdk.core.util import times
def CheckResponseRootTypeHook(version='v1'):
    """Raises an exception if the response is not a root ca."""

    def CheckResponseRootTypeHookVersioned(response, unused_args):
        resource_args.CheckExpectedCAType(base.GetMessagesModule(api_version=version).CertificateAuthority.TypeValueValuesEnum.SELF_SIGNED, response, version=version)
        return response
    return CheckResponseRootTypeHookVersioned