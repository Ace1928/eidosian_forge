from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateDatabaseDdlRequest(_messages.Message):
    """Enqueues the given DDL statements to be applied, in order but not
  necessarily all at once, to the database schema at some point (or points) in
  the future. The server checks that the statements are executable
  (syntactically valid, name tables that exist, etc.) before enqueueing them,
  but they may still fail upon later execution (e.g., if a statement from
  another batch of statements is applied first and it conflicts in some way,
  or if there is some data-related problem like a `NULL` value in a column to
  which `NOT NULL` would be added). If a statement fails, all subsequent
  statements in the batch are automatically cancelled. Each batch of
  statements is assigned a name which can be used with the Operations API to
  monitor progress. See the operation_id field for more details.

  Fields:
    operationId: If empty, the new update request is assigned an
      automatically-generated operation ID. Otherwise, `operation_id` is used
      to construct the name of the resulting Operation. Specifying an explicit
      operation ID simplifies determining whether the statements were executed
      in the event that the UpdateDatabaseDdl call is replayed, or the return
      value is otherwise lost: the database and `operation_id` fields can be
      combined to form the name of the resulting longrunning.Operation:
      `/operations/`. `operation_id` should be unique within the database, and
      must be a valid identifier: `a-z*`. Note that automatically-generated
      operation IDs always begin with an underscore. If the named operation
      already exists, UpdateDatabaseDdl returns `ALREADY_EXISTS`.
    protoDescriptors: Optional. Proto descriptors used by CREATE/ALTER PROTO
      BUNDLE statements. Contains a protobuf-serialized [google.protobuf.FileD
      escriptorSet](https://github.com/protocolbuffers/protobuf/blob/main/src/
      google/protobuf/descriptor.proto). To generate it,
      [install](https://grpc.io/docs/protoc-installation/) and run `protoc`
      with --include_imports and --descriptor_set_out. For example, to
      generate for moon/shot/app.proto, run ``` $protoc --proto_path=/app_path
      --proto_path=/lib_path \\ --include_imports \\
      --descriptor_set_out=descriptors.data \\ moon/shot/app.proto ``` For more
      details, see protobuffer [self
      description](https://developers.google.com/protocol-
      buffers/docs/techniques#self-description).
    statements: Required. DDL statements to be applied to the database.
  """
    operationId = _messages.StringField(1)
    protoDescriptors = _messages.BytesField(2)
    statements = _messages.StringField(3, repeated=True)