from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def ConstructFailoverConfig(threshold):
    return messages.ServiceLbPolicyFailoverConfig(failoverHealthThreshold=threshold)