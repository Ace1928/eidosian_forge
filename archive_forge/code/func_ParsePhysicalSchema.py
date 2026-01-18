from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from os import path
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from googlecloudsdk.api_lib.data_catalog import entries_v1
from googlecloudsdk.api_lib.data_catalog import util as api_util
from googlecloudsdk.command_lib.concepts import exceptions as concept_exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def ParsePhysicalSchema(ref, args, request):
    """Parses physical schema from file after obtaining information about its type.

  Args:
    ref: The entry resource reference.
    args: The parsed args namespace.
    request: The update entry request.

  Returns:
    Request with merged GCS file pattern.

  Raises:
    InvalidPhysicalSchemaError: if physical schema type is unknown
  """
    if not args.IsSpecified('physical_schema_type'):
        return request
    del ref
    client = entries_v1.EntriesClient()
    messages = client.messages
    if args.IsSpecified('physical_schema_file'):
        schema_abs_path = path.expanduser(args.physical_schema_file)
        schema_text = files.ReadFileContents(schema_abs_path)
    else:
        schema_text = ''
    schema_type = args.physical_schema_type
    if schema_type == 'avro':
        schema = messages.GoogleCloudDatacatalogV1PhysicalSchemaAvroSchema()
        schema.text = schema_text
    elif schema_type == 'thrift':
        schema = messages.GoogleCloudDatacatalogV1PhysicalSchemaThriftSchema()
        schema.text = schema_text
    elif schema_type == 'protobuf':
        schema = messages.GoogleCloudDatacatalogV1PhysicalSchemaProtobufSchema()
        schema.text = schema_text
    elif schema_type == 'parquet':
        schema = messages.GoogleCloudDatacatalogV1PhysicalSchemaParquetSchema()
    elif schema_type == 'orc':
        schema = messages.GoogleCloudDatacatalogV1PhysicalSchemaOrcSchema()
    else:
        raise InvalidPhysicalSchemaError('Unknown physical schema type. Must be one of: avro, thrift, protobuf,parquet, orc')
    arg_utils.SetFieldInMessage(request, 'googleCloudDatacatalogV1Entry.schema.physicalSchema.' + schema_type, schema)
    return request