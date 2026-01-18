from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.routers import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def ConvertRouteType(self, route_type):
    if route_type == 'IMPORT':
        return 'ROUTE_POLICY_TYPE_IMPORT'
    elif route_type == 'EXPORT':
        return 'ROUTE_POLICY_TYPE_EXPORT'
    else:
        return route_type