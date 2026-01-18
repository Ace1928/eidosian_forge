from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
Lookup membership name for a specific pair of member key id and group email.

  Args:
    version: Release track information
    group_id: str, group id (e.g. groups/03qco8b4452k99t)
    member_email: str, member email
  Returns:
    LookupMembershipNameResponse: Response message for LookupMembershipName
    operation which is containing a resource name of the membership in the
    format:
    'name: members/{member_id}'
  