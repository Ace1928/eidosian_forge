from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageDescriptor(_messages.Message):
    """Contains information about how a table's data is stored and accessed by
  open source query engines.

  Fields:
    inputFormat: Optional. Specifies the fully qualified class name of the
      InputFormat (e.g. "org.apache.hadoop.hive.ql.io.orc.OrcInputFormat").
      The maximum length is 128 characters.
    locationUri: Optional. The physical location of the table (e.g.
      'gs://spark-dataproc-data/pangea-data/case_sensitive/' or 'gs://spark-
      dataproc-data/pangea-data/*'). The maximum length is 2056 bytes.
    outputFormat: Optional. Specifies the fully qualified class name of the
      OutputFormat (e.g. "org.apache.hadoop.hive.ql.io.orc.OrcOutputFormat").
      The maximum length is 128 characters.
    serdeInfo: Optional. Serializer and deserializer information.
  """
    inputFormat = _messages.StringField(1)
    locationUri = _messages.StringField(2)
    outputFormat = _messages.StringField(3)
    serdeInfo = _messages.MessageField('SerDeInfo', 4)