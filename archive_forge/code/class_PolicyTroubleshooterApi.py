from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
class PolicyTroubleshooterApi(object):
    """Base Class for Policy Troubleshooter API."""

    def __new__(cls, release_track):
        if release_track == base.ReleaseTrack.ALPHA:
            return super(PolicyTroubleshooterApi, cls).__new__(PolicyTroubleshooterApiAlpha)
        if release_track == base.ReleaseTrack.BETA:
            return super(PolicyTroubleshooterApi, cls).__new__(PolicyTroubleshooterApiBeta)
        if release_track == base.ReleaseTrack.GA:
            return super(PolicyTroubleshooterApi, cls).__new__(PolicyTroubleshooterApiGA)

    def __init__(self, release_track):
        api_version = GetApiVersion(release_track)
        self.client = apis.GetClientInstance(_API_NAME, api_version)
        self.messages = apis.GetMessagesModule(_API_NAME, api_version)

    @abc.abstractmethod
    def TroubleshootIAMPolicies(self, access_tuple):
        pass

    @abc.abstractmethod
    def GetPolicyTroubleshooterAccessTuple(self, condition_context=None, full_resource_name=None, principal_email=None, permission=None):
        pass

    @abc.abstractmethod
    def GetPolicyTroubleshooterConditionContext(self, destination=None, request=None, resource=None):
        pass

    @abc.abstractmethod
    def GetPolicyTroubleshooterPeer(self, destination_ip=None, destination_port=None):
        pass

    @abc.abstractmethod
    def GetPolicyTroubleshooterRequest(self, request_time=None):
        pass

    @abc.abstractmethod
    def GetPolicyTroubleshooterResource(self, resource_name=None, resource_service=None, resource_type=None):
        pass