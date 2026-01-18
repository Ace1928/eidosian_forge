from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def SetContentUpdateMask(unused_ref, args, request):
    if args.bucket:
        request.updateMask = 'bucketName'
    elif args.uiconfig:
        request.updateMask = 'uiconfig'
    return request