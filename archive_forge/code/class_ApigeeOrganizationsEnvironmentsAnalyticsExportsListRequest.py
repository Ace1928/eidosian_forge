from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsAnalyticsExportsListRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsAnalyticsExportsListRequest object.

  Fields:
    parent: Required. Names of the parent organization and environment. Must
      be of the form `organizations/{org}/environments/{env}`.
  """
    parent = _messages.StringField(1, required=True)