from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_templates import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
@staticmethod
def GetViewEnumValue(view, request_message):
    if view == 'FULL':
        return request_message.ViewValueValuesEnum.FULL
    elif view == 'BASIC':
        return request_message.ViewValueValuesEnum.BASIC
    return None