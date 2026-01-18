from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SparkHistoryServer(_messages.Message):
    """Spark History Server.

  Fields:
    sparkHistoryServerConfig: Optional. Spark History Server configurations
      for a given version.
  """
    sparkHistoryServerConfig = _messages.MessageField('SparkHistoryServerConfig', 1)