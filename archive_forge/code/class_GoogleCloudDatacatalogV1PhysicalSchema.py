from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1PhysicalSchema(_messages.Message):
    """Native schema used by a resource represented as an entry. Used by query
  engines for deserializing and parsing source data.

  Fields:
    avro: Schema in Avro JSON format.
    csv: Marks a CSV-encoded data source.
    orc: Marks an ORC-encoded data source.
    parquet: Marks a Parquet-encoded data source.
    protobuf: Schema in protocol buffer format.
    thrift: Schema in Thrift format.
  """
    avro = _messages.MessageField('GoogleCloudDatacatalogV1PhysicalSchemaAvroSchema', 1)
    csv = _messages.MessageField('GoogleCloudDatacatalogV1PhysicalSchemaCsvSchema', 2)
    orc = _messages.MessageField('GoogleCloudDatacatalogV1PhysicalSchemaOrcSchema', 3)
    parquet = _messages.MessageField('GoogleCloudDatacatalogV1PhysicalSchemaParquetSchema', 4)
    protobuf = _messages.MessageField('GoogleCloudDatacatalogV1PhysicalSchemaProtobufSchema', 5)
    thrift = _messages.MessageField('GoogleCloudDatacatalogV1PhysicalSchemaThriftSchema', 6)