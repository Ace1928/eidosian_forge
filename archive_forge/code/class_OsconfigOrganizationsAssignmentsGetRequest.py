from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsconfigOrganizationsAssignmentsGetRequest(_messages.Message):
    """A OsconfigOrganizationsAssignmentsGetRequest object.

  Fields:
    name: The resource name of the Assignment.
  """
    name = _messages.StringField(1, required=True)