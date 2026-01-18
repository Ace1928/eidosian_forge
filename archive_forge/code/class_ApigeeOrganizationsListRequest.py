from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsListRequest(_messages.Message):
    """A ApigeeOrganizationsListRequest object.

  Fields:
    parent: Required. Use the following structure in your request:
      `organizations`
  """
    parent = _messages.StringField(1, required=True)