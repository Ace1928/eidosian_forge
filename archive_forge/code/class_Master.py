from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Master(_messages.Message):
    """Master is the configuration for components on master.

  Fields:
    signalsConfig: Configuration used to enable sending selected master logs
      and metrics to customer project. This feature is has been replaced by
      the system component options in Cluster.logging_config.component_config.
  """
    signalsConfig = _messages.MessageField('MasterSignalsConfig', 1)