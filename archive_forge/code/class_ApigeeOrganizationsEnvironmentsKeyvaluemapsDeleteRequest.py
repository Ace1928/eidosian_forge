from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsKeyvaluemapsDeleteRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsKeyvaluemapsDeleteRequest object.

  Fields:
    name: Required. Name of the key value map. Use the following structure in
      your request:
      `organizations/{org}/environments/{env}/keyvaluemaps/{keyvaluemap}`
  """
    name = _messages.StringField(1, required=True)