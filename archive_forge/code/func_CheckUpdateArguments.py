from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.fleet import client
from googlecloudsdk.command_lib.util.args import labels_util
def CheckUpdateArguments(ref, args, request):
    del ref, args
    if request.updateMask is None or not request.updateMask:
        request.updateMask = 'name'
    return request