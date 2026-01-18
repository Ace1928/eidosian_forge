from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.binauthz import apis
from googlecloudsdk.command_lib.iam import iam_util
def AddBinding(self, any_ref, member, role):
    """Does an atomic Read-Modify-Write, adding the member to the role."""
    policy = self.Get(any_ref)
    iam_util.AddBindingToIamPolicy(self.messages.Binding, policy, member, role)
    return self.Set(any_ref, policy)