from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import exceptions
def UpdateOperationRequestNameVariable(unused_ref, unused_args, request):
    request.name += '/locations/global'
    return request