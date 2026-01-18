from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AwsOrganization(_messages.Message):
    """An organization is a collection of accounts that are centrally managed
  together using consolidated billing, organized hierarchically with
  organizational units (OUs), and controlled with policies.

  Fields:
    id: The unique identifier (ID) for the organization. The regex pattern for
      an organization ID string requires "o-" followed by from 10 to 32
      lowercase letters or digits.
  """
    id = _messages.StringField(1)