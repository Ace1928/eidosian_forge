from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GetDatabaseDdlResponse(_messages.Message):
    """The response for GetDatabaseDdl.

  Fields:
    protoDescriptors: Proto descriptors stored in the database. Contains a
      protobuf-serialized [google.protobuf.FileDescriptorSet](https://github.c
      om/protocolbuffers/protobuf/blob/main/src/google/protobuf/descriptor.pro
      to). For more details, see protobuffer [self
      description](https://developers.google.com/protocol-
      buffers/docs/techniques#self-description).
    statements: A list of formatted DDL statements defining the schema of the
      database specified in the request.
  """
    protoDescriptors = _messages.BytesField(1)
    statements = _messages.StringField(2, repeated=True)