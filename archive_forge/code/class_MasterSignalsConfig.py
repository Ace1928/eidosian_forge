from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MasterSignalsConfig(_messages.Message):
    """MasterSignalsConfig is the configuration for exposing selected master
  logs and metrics to customer

  Enums:
    LogEnabledComponentsValueListEntryValuesEnum:

  Fields:
    enableMetrics: Enable sendings metrics to customer
    logEnabledComponents: Select components to expose logs
  """

    class LogEnabledComponentsValueListEntryValuesEnum(_messages.Enum):
        """LogEnabledComponentsValueListEntryValuesEnum enum type.

    Values:
      COMPONENT_UNSPECIFIED: Didn't specify any components. Used to avoid
        overriding existing list.
      APISERVER: kube-apiserver
      SCHEDULER: kube-scheduler
      CONTROLLER_MANAGER: kube-controller-manager
      ADDON_MANAGER: kube-addon-manager
    """
        COMPONENT_UNSPECIFIED = 0
        APISERVER = 1
        SCHEDULER = 2
        CONTROLLER_MANAGER = 3
        ADDON_MANAGER = 4
    enableMetrics = _messages.BooleanField(1)
    logEnabledComponents = _messages.EnumField('LogEnabledComponentsValueListEntryValuesEnum', 2, repeated=True)