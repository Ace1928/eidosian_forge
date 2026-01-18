from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.oslogin import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.oslogin import oslogin_utils
from googlecloudsdk.core import properties
def _TransformExpiry(resource, undefined=None):
    display = None
    value = resource.get('value')
    if value:
        display = oslogin_utils.ConvertUsecToRfc3339(value.get('expirationTimeUsec'))
    return display or undefined