from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.active_directory import exceptions
from googlecloudsdk.command_lib.active_directory import util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core.console import console_io
def ReadPfxPasswordIfNeeded(unused_ref, args, request):
    """Allows for the pfx password to be read from stdin if not specified."""
    del unused_ref
    if args.certificate_password:
        request.lDAPSSettings.certificatePassword = args.certificate_password
        return request
    if args.clear_certificates:
        return request
    secret = GetPfxPasssword()
    request.lDAPSSettings.certificatePassword = secret
    return request