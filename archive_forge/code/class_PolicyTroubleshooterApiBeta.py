from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
class PolicyTroubleshooterApiBeta(PolicyTroubleshooterApi):
    """Base Class for Policy Troubleshooter API Beta."""

    def TroubleshootIAMPolicies(self, access_tuple):
        request = self.messages.GoogleCloudPolicytroubleshooterIamV3betaTroubleshootIamPolicyRequest(accessTuple=access_tuple)
        return self.client.iam.Troubleshoot(request)

    def GetPolicyTroubleshooterAccessTuple(self, condition_context=None, full_resource_name=None, principal_email=None, permission=None):
        return self.messages.GoogleCloudPolicytroubleshooterIamV3betaAccessTuple(fullResourceName=full_resource_name, principal=principal_email, permission=permission, conditionContext=condition_context)

    def GetPolicyTroubleshooterRequest(self, request_time=None):
        return self.messages.GoogleCloudPolicytroubleshooterIamV3betaConditionContextRequest(receiveTime=request_time)

    def GetPolicyTroubleshooterResource(self, resource_name=None, resource_service=None, resource_type=None):
        return self.messages.GoogleCloudPolicytroubleshooterIamV3betaConditionContextResource(name=resource_name, service=resource_service, type=resource_type)

    def GetPolicyTroubleshooterPeer(self, destination_ip=None, destination_port=None):
        return self.messages.GoogleCloudPolicytroubleshooterIamV3betaConditionContextPeer(ip=destination_ip, port=destination_port)

    def GetPolicyTroubleshooterConditionContext(self, destination=None, request=None, resource=None):
        return self.messages.GoogleCloudPolicytroubleshooterIamV3betaConditionContext(destination=destination, request=request, resource=resource)