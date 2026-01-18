from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import mimetypes
import os
from apitools.base.py import exceptions as api_exceptions
from apitools.base.py import list_pager
from apitools.base.py import transfer
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import exceptions as http_exc
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import exceptions as core_exc
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import transports
from googlecloudsdk.core.util import scaled_integer
import six
def AddIamPolicyBindings(self, bucket_ref, member_roles):
    """Add IAM policy bindings on the specified bucket.

    Does an atomic Read-Modify-Write, adding the member to the role.

    Args:
      bucket_ref: storage_util.BucketReference to the bucket with the policy.
      member_roles: List of 2-tuples in the form [(member, role), ...].

    Returns:
      The new IAM Policy.
    """
    policy = self.GetIamPolicy(bucket_ref)
    policy.version = iam_util.MAX_LIBRARY_IAM_SUPPORTED_VERSION
    policy_was_updated = False
    for member, role in member_roles:
        if iam_util.AddBindingToIamPolicy(self.messages.Policy.BindingsValueListEntry, policy, member, role):
            policy_was_updated = True
    if policy_was_updated:
        return self.SetIamPolicy(bucket_ref, policy)
    return policy