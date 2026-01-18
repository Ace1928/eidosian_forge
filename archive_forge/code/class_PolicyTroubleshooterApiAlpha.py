from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
class PolicyTroubleshooterApiAlpha(PolicyTroubleshooterApi):
    """Base Class for Policy Troubleshooter API Alpha."""

    def TroubleshootIAMPolicies(self, access_tuple):
        request = self.messages.GoogleCloudPolicytroubleshooterIamV3alphaTroubleshootIamPolicyRequest(accessTuple=access_tuple)
        return self.client.iam.Troubleshoot(request)

    def GetPolicyTroubleshooterAccessTuple(self, condition_context=None, full_resource_name=None, principal_email=None, permission=None):
        return self.messages.GoogleCloudPolicytroubleshooterIamV3alphaAccessTuple(fullResourceName=full_resource_name, principal=principal_email, permission=permission, conditionContext=condition_context)

    def GetPolicyTroubleshooterRequest(self, request_time=None):
        return self.messages.GoogleCloudPolicytroubleshooterIamV3alphaConditionContextRequest(receiveTime=request_time)

    def GetPolicyTroubleshooterResource(self, resource_name=None, resource_service=None, resource_type=None):
        return self.messages.GoogleCloudPolicytroubleshooterIamV3alphaConditionContextResource(name=resource_name, service=resource_service, type=resource_type)

    def GetPolicyTroubleshooterPeer(self, destination_ip=None, destination_port=None):
        return self.messages.GoogleCloudPolicytroubleshooterIamV3alphaConditionContextPeer(ip=destination_ip, port=destination_port)

    def GetPolicyTroubleshooterConditionContext(self, destination=None, request=None, resource=None):
        return self.messages.GoogleCloudPolicytroubleshooterIamV3alphaConditionContext(destination=destination, request=request, resource=resource)