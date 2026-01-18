from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
def GetPolicyTroubleshooterAccessTuple(self, condition_context=None, full_resource_name=None, principal_email=None, permission=None):
    return self.messages.GoogleCloudPolicytroubleshooterIamV3AccessTuple(fullResourceName=full_resource_name, principal=principal_email, permission=permission, conditionContext=condition_context)