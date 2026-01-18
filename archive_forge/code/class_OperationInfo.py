from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OperationInfo(_messages.Message):
    """A message representing the message types used by a long-running
  operation. Example: rpc Export(ExportRequest) returns
  (google.longrunning.Operation) { option (google.longrunning.operation_info)
  = { response_type: "ExportResponse" metadata_type: "ExportMetadata" }; }

  Fields:
    metadataType: Required. The message name of the metadata type for this
      long-running operation. If the response is in a different package from
      the rpc, a fully-qualified message name must be used (e.g.
      `google.protobuf.Struct`). Note: Altering this value constitutes a
      breaking change.
    responseType: Required. The message name of the primary return type for
      this long-running operation. This type will be used to deserialize the
      LRO's response. If the response is in a different package from the rpc,
      a fully-qualified message name must be used (e.g.
      `google.protobuf.Struct`). Note: Altering this value constitutes a
      breaking change.
  """
    metadataType = _messages.StringField(1)
    responseType = _messages.StringField(2)