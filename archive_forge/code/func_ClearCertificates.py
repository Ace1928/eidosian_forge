from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.active_directory import exceptions
from googlecloudsdk.command_lib.active_directory import util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core.console import console_io
def ClearCertificates(unused_ref, args, request):
    del unused_ref
    if args.clear_certificates:
        request = AddFieldToUpdateMask('certificatePfx', request)
        request = AddFieldToUpdateMask('certificatePassword', request)
    return request