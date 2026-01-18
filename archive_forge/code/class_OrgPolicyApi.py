from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.api_lib.orgpolicy import utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.generated_clients.apis.orgpolicy.v2 import orgpolicy_v2_messages
class OrgPolicyApi(object):
    """Base class for Org Policy API."""

    def __new__(cls, release_track):
        if release_track == base.ReleaseTrack.GA:
            return super(OrgPolicyApi, cls).__new__(OrgPolicyApiGA)

    def __init__(self, release_track):
        api_version = GetApiVersion(release_track)
        self.client = apis.GetClientInstance(ORG_POLICY_API_NAME, api_version)
        self.messages = apis.GetMessagesModule(ORG_POLICY_API_NAME, api_version)

    @abc.abstractmethod
    def GetPolicy(self, name):
        pass

    @abc.abstractmethod
    def GetEffectivePolicy(self, name):
        pass

    @abc.abstractmethod
    def DeletePolicy(self, name, etag=None) -> orgpolicy_v2_messages.GoogleProtobufEmpty:
        pass

    @abc.abstractmethod
    def ListPolicies(self, parent):
        pass

    @abc.abstractmethod
    def ListConstraints(self, parent):
        pass

    @abc.abstractmethod
    def CreatePolicy(self, policy):
        pass

    @abc.abstractmethod
    def UpdatePolicy(self, policy, update_mask=None):
        pass

    @abc.abstractmethod
    def CreateCustomConstraint(self, custom_constraint):
        pass

    @abc.abstractmethod
    def UpdateCustomConstraint(self, custom_constraint):
        pass

    @abc.abstractmethod
    def GetCustomConstraint(self, name):
        pass

    @abc.abstractmethod
    def DeleteCustomConstraint(self, name):
        pass

    @abc.abstractmethod
    def CreateEmptyPolicySpec(self):
        pass

    @abc.abstractmethod
    def BuildPolicy(self, name):
        pass

    @abc.abstractmethod
    def BuildEmptyPolicy(self, name, has_spec=False, has_dry_run_spec=False):
        pass

    @abc.abstractmethod
    def BuildPolicySpecPolicyRule(self, condition=None, allow_all=None, deny_all=None, enforce=None, values=None):
        pass

    @abc.abstractmethod
    def BuildPolicySpecPolicyRuleStringValues(self, allowed_values=(), denied_values=()):
        pass