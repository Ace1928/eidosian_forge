from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsDashboardsCreateRequest(_messages.Message):
    """A MonitoringProjectsDashboardsCreateRequest object.

  Fields:
    dashboard: A Dashboard resource to be passed as the request body.
    parent: Required. The project on which to execute the request. The format
      is: projects/[PROJECT_ID_OR_NUMBER] The [PROJECT_ID_OR_NUMBER] must
      match the dashboard resource name.
    validateOnly: If set, validate the request and preview the review, but do
      not actually save it.
  """
    dashboard = _messages.MessageField('Dashboard', 1)
    parent = _messages.StringField(2, required=True)
    validateOnly = _messages.BooleanField(3)