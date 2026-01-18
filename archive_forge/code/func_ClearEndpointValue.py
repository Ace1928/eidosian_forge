from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import exceptions
def ClearEndpointValue(endpoint, endpoint_name):
    proto_endpoint_fields = {'cloudFunction', 'appEngineVersion', 'cloudRunRevision'}
    if endpoint_name in proto_endpoint_fields:
        setattr(endpoint, endpoint_name, None)
    else:
        setattr(endpoint, endpoint_name, '')