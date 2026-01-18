from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.api_lib.orgpolicy import utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.generated_clients.apis.orgpolicy.v2 import orgpolicy_v2_messages
def BuildPolicySpecPolicyRule(self, condition=None, allow_all=None, deny_all=None, enforce=None, values=None):
    return self.messages.GoogleCloudOrgpolicyV2PolicySpecPolicyRule(condition=condition, allowAll=allow_all, denyAll=deny_all, enforce=enforce, values=values)