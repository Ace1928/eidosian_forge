from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuxiliaryServicesConfig(_messages.Message):
    """Auxiliary Service Configs.

  Fields:
    sparkHistoryServer: Optional. Spark History Servor message.
  """
    sparkHistoryServer = _messages.MessageField('SparkHistoryServer', 1)