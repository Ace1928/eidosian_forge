from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WriteSessionSparkApplicationContextRequest(_messages.Message):
    """Write Spark Application data to internal storage systems

  Fields:
    parent: Required. Parent (Batch) resource reference.
    sparkWrapperObjects: Required. The batch of spark application context
      objects sent for ingestion.
  """
    parent = _messages.StringField(1)
    sparkWrapperObjects = _messages.MessageField('SparkWrapperObject', 2, repeated=True)