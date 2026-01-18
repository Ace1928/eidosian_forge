from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigManagementPolicyControllerMonitoring(_messages.Message):
    """PolicyControllerMonitoring specifies the backends Policy Controller
  should export metrics to. For example, to specify metrics should be exported
  to Cloud Monitoring and Prometheus, specify backends: ["cloudmonitoring",
  "prometheus"]

  Enums:
    BackendsValueListEntryValuesEnum:

  Fields:
    backends: Specifies the list of backends Policy Controller will export to.
      An empty list would effectively disable metrics export.
  """

    class BackendsValueListEntryValuesEnum(_messages.Enum):
        """BackendsValueListEntryValuesEnum enum type.

    Values:
      MONITORING_BACKEND_UNSPECIFIED: Backend cannot be determined
      PROMETHEUS: Prometheus backend for monitoring
      CLOUD_MONITORING: Stackdriver/Cloud Monitoring backend for monitoring
    """
        MONITORING_BACKEND_UNSPECIFIED = 0
        PROMETHEUS = 1
        CLOUD_MONITORING = 2
    backends = _messages.EnumField('BackendsValueListEntryValuesEnum', 1, repeated=True)