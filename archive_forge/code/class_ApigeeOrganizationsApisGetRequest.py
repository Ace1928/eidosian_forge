from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsApisGetRequest(_messages.Message):
    """A ApigeeOrganizationsApisGetRequest object.

  Fields:
    name: Required. Name of the API proxy in the following format:
      `organizations/{org}/apis/{api}`
  """
    name = _messages.StringField(1, required=True)