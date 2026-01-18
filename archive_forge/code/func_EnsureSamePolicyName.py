from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
import os
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.routers import flags
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def EnsureSamePolicyName(self, policy_name, route_policy):
    if policy_name is not None and hasattr(route_policy, 'name'):
        if policy_name != route_policy.name:
            raise exceptions.BadArgumentException('policy-name', 'The policy name provided [{0}] does not match the one from the file [{1}]'.format(policy_name, route_policy.name))
    if not hasattr(route_policy, 'name') and policy_name is not None:
        route_policy.name = policy_name