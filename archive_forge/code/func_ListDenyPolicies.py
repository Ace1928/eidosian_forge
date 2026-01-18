from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import binascii
from apitools.base.protorpclite import messages as apitools_messages
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import yaml
import six
def ListDenyPolicies(resource_id, resource_type, release_track):
    """Gets the IAM Deny policies for an organization.

  Args:
    resource_id: id for the resource
    resource_type: what type of a resource the id represents. Either
      organization, project, or folder
    release_track: ALPHA or BETA or GA

  Returns:
    The output from the ListPolicies API call for deny policies for the passed
    resource.
  """
    client = GetClientInstance(release_track)
    messages = GetMessagesModule(release_track)
    policies_to_return = []
    if resource_type in ['organization', 'folder', 'project']:
        attachment_point = 'policies/cloudresourcemanager.googleapis.com%2F{}s%2F{}/denypolicies'.format(resource_type, resource_id)
        policies_to_fetch = client.policies.ListPolicies(messages.IamPoliciesListPoliciesRequest(parent=attachment_point)).policies
        for policy_metadata in policies_to_fetch:
            policy = client.policies.Get(messages.IamPoliciesGetRequest(name=policy_metadata.name))
            policies_to_return.append(policy)
        return policies_to_return
    raise gcloud_exceptions.UnknownArgumentException('resource_type', resource_type)