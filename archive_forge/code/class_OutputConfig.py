from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OutputConfig(_messages.Message):
    """Output configuration for export assets destination.

  Fields:
    bigqueryDestination: Destination on BigQuery. The output table stores the
      fields in asset Protobuf as columns in BigQuery.
    gcsDestination: Destination on Cloud Storage.
  """
    bigqueryDestination = _messages.MessageField('BigQueryDestination', 1)
    gcsDestination = _messages.MessageField('GcsDestination', 2)