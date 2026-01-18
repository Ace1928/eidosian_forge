from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetParamsForRegistry(version, args, parent=None):
    params = {'project': properties.VALUES.core.project.GetOrFail}
    if version == 'v2':
        params['location'] = args.location
    if parent is not None:
        if parent == 'managedZones':
            params['managedZone'] = args.zone
        if parent == 'responsePolicies':
            params['responsePolicy'] = args.response_policy
    return params