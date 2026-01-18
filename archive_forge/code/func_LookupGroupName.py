from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
def LookupGroupName(version, email):
    """Lookup Group Name for a specified group key id.

  Args:
    version: Release track information
    email: str, group email

  Returns:
    LookupGroupNameResponse: Response message for LookupGroupName operation
    which is containing a resource name of the group in the format:
    'name: groups/{group_id}'
  """
    client = GetClient(version)
    messages = GetMessages(version)
    encoding.AddCustomJsonFieldMapping(messages.CloudidentityGroupsLookupRequest, 'groupKey_id', 'groupKey.id')
    return client.groups.Lookup(messages.CloudidentityGroupsLookupRequest(groupKey_id=email))