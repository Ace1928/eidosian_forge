from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.util import times
def ResetDefaultMaskField(unused_instance_ref, unused_args, req):
    req.updateMask = ''
    return req