from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsGetAddonsConfigRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsGetAddonsConfigRequest object.

  Fields:
    name: Required. Name of the add-ons config. Must be in the format of
      `/organizations/{org}/environments/{env}/addonsConfig`
  """
    name = _messages.StringField(1, required=True)