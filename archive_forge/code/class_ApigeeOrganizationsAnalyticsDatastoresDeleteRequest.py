from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsAnalyticsDatastoresDeleteRequest(_messages.Message):
    """A ApigeeOrganizationsAnalyticsDatastoresDeleteRequest object.

  Fields:
    name: Required. Resource name of the Datastore to be deleted. Must be of
      the form `organizations/{org}/analytics/datastores/{datastoreId}`
  """
    name = _messages.StringField(1, required=True)