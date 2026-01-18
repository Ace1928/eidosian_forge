from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import types
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
def DefineIamPolicyFunctions(self):
    """Defines all of the IAM functionality on the calling class."""

    def GetIamPolicy(self, object_ref):
        """Gets an IAM Policy on an object.

      Args:
        self: The self of the class this is set on.
        object_ref: Resource, reference for object IAM policy belongs to.

      Returns:
        The IAM policy.
      """
        req = self.get_iam_policy_request(resource=object_ref.RelativeName())
        return self.service.GetIamPolicy(req)

    def SetIamPolicy(self, object_ref, policy, update_mask=None):
        """Sets an IAM Policy on an object.

      Args:
        self: The self of the class this is set on.
        object_ref: Resource, reference for object IAM policy belongs to.
        policy: the policy to be set.
        update_mask: fields being update on the IAM policy.

      Returns:
        The IAM policy.
      """
        policy_request = self.messages.ApigatewaySetIamPolicyRequest(policy=policy, updateMask=update_mask)
        req = self.set_iam_policy_request(apigatewaySetIamPolicyRequest=policy_request, resource=object_ref.RelativeName())
        return self.service.SetIamPolicy(req)

    def AddIamPolicyBinding(self, object_ref, member, role):
        """Adds an IAM role to a member on an object.

      Args:
        self: The self of the class this is set on.
        object_ref: Resource, reference for object IAM policy belongs to.
        member: the member the binding is being added to.
        role: the role which to bind to the member.

      Returns:
        The IAM policy.
      """
        policy = self.GetIamPolicy(object_ref)
        iam_util.AddBindingToIamPolicy(self.messages.ApigatewayBinding, policy, member, role)
        return self.SetIamPolicy(object_ref, policy, 'bindings,etag')

    def RemoveIamPolicyBinding(self, object_ref, member, role):
        """Adds an IAM role for a member on an object.

      Args:
        self: The self of the class this is set on
        object_ref: Resource, reference for object IAM policy belongs to
        member: the member the binding is removed for
        role: the role which is being removed from the member

      Returns:
        The IAM policy
      """
        policy = self.GetIamPolicy(object_ref)
        iam_util.RemoveBindingFromIamPolicy(policy, member, role)
        return self.SetIamPolicy(object_ref, policy, 'bindings,etag')
    setattr(self, 'GetIamPolicy', types.MethodType(GetIamPolicy, self))
    setattr(self, 'SetIamPolicy', types.MethodType(SetIamPolicy, self))
    setattr(self, 'AddIamPolicyBinding', types.MethodType(AddIamPolicyBinding, self))
    setattr(self, 'RemoveIamPolicyBinding', types.MethodType(RemoveIamPolicyBinding, self))