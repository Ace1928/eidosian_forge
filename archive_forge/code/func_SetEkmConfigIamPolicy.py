from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import base
from googlecloudsdk.command_lib.iam import iam_util
def SetEkmConfigIamPolicy(ekm_config_name, policy, update_mask):
    """Set the IAM Policy attached to the named EkmConfig to the given policy.

  If 'policy' has no etag specified, this will BLINDLY OVERWRITE the IAM policy!

  Args:
      ekm_config_name:  A string name of the EkmConfig.
      policy: An apitools wrapper for the IAM Policy.
      update_mask: str, FieldMask represented as comma-separated field names.

  Returns:
      The IAM Policy.
  """
    client = base.GetClientInstance()
    messages = base.GetMessagesModule()
    policy.version = iam_util.MAX_LIBRARY_IAM_SUPPORTED_VERSION
    if not update_mask:
        update_mask = 'version'
    elif 'version' not in update_mask:
        update_mask += ',version'
    req = messages.CloudkmsProjectsLocationsEkmConfigSetIamPolicyRequest(resource=ekm_config_name, setIamPolicyRequest=messages.SetIamPolicyRequest(policy=policy, updateMask=update_mask))
    return client.projects_locations_ekmConfig.SetIamPolicy(req)