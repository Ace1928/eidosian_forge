from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsDashboardsGetRequest(_messages.Message):
    """A MonitoringProjectsDashboardsGetRequest object.

  Fields:
    name: Required. The resource name of the Dashboard. The format is one of:
      dashboards/[DASHBOARD_ID] (for system dashboards)
      projects/[PROJECT_ID_OR_NUMBER]/dashboards/[DASHBOARD_ID] (for custom
      dashboards).
  """
    name = _messages.StringField(1, required=True)