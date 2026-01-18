from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.app import appengine_api_client
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
import six
class IapIamResource(six.with_metaclass(abc.ABCMeta, object)):
    """Base class for IAP IAM resources."""

    def __init__(self, release_track, project):
        """Base Constructor for an IAP IAM resource.

    Args:
      release_track: base.ReleaseTrack, release track of command.
      project: Project of the IAP IAM resource
    """
        self.release_track = release_track
        self.api_version = _ApiVersion(release_track)
        self.client = apis.GetClientInstance(IAP_API, self.api_version)
        self.registry = _GetRegistry(self.api_version)
        self.project = project

    @property
    def messages(self):
        return self.client.MESSAGES_MODULE

    @property
    def service(self):
        return getattr(self.client, self.api_version)

    @abc.abstractmethod
    def _Name(self):
        """Human-readable name of the resource."""
        pass

    @abc.abstractmethod
    def _Parse(self):
        """Parses the IAP IAM resource from the arguments."""
        pass

    def _GetIamPolicy(self, resource_ref):
        request = self.messages.IapGetIamPolicyRequest(resource=resource_ref.RelativeName(), getIamPolicyRequest=self.messages.GetIamPolicyRequest(options=self.messages.GetPolicyOptions(requestedPolicyVersion=iam_util.MAX_LIBRARY_IAM_SUPPORTED_VERSION)))
        return self.service.GetIamPolicy(request)

    def GetIamPolicy(self):
        """Get IAM policy for an IAP IAM resource."""
        resource_ref = self._Parse()
        return self._GetIamPolicy(resource_ref)

    def _SetIamPolicy(self, resource_ref, policy):
        policy.version = iam_util.MAX_LIBRARY_IAM_SUPPORTED_VERSION
        request = self.messages.IapSetIamPolicyRequest(resource=resource_ref.RelativeName(), setIamPolicyRequest=self.messages.SetIamPolicyRequest(policy=policy))
        response = self.service.SetIamPolicy(request)
        iam_util.LogSetIamPolicy(resource_ref.RelativeName(), self._Name())
        return response

    def SetIamPolicy(self, policy_file):
        """Set the IAM policy for an IAP IAM resource."""
        policy = iam_util.ParsePolicyFile(policy_file, self.messages.Policy)
        resource_ref = self._Parse()
        return self._SetIamPolicy(resource_ref, policy)

    def AddIamPolicyBinding(self, member, role, condition):
        """Add IAM policy binding to an IAP IAM resource."""
        resource_ref = self._Parse()
        policy = self._GetIamPolicy(resource_ref)
        iam_util.AddBindingToIamPolicyWithCondition(self.messages.Binding, self.messages.Expr, policy, member, role, condition)
        self._SetIamPolicy(resource_ref, policy)

    def RemoveIamPolicyBinding(self, member, role, condition, all_conditions):
        """Remove IAM policy binding from an IAP IAM resource."""
        resource_ref = self._Parse()
        policy = self._GetIamPolicy(resource_ref)
        iam_util.RemoveBindingFromIamPolicyWithCondition(policy, member, role, condition, all_conditions=all_conditions)
        self._SetIamPolicy(resource_ref, policy)