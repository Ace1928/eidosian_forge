from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import datetime
import functools
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple
from absl import app
from absl import flags
import yaml
import table_formatter
import bq_utils
from clients import utils as bq_client_utils
from utils import bq_error
from utils import bq_id_utils
from pyglib import stringutil
def CreateExternalTableDefinition(source_format, source_uris, schema, autodetect, connection_id=None, ignore_unknown_values=False, hive_partitioning_mode=None, hive_partitioning_source_uri_prefix=None, require_hive_partition_filter=None, use_avro_logical_types=False, parquet_enum_as_string=False, parquet_enable_list_inference=False, metadata_cache_mode=None, object_metadata=None, preserve_ascii_control_characters=False, reference_file_schema_uri=None, encoding=None, file_set_spec_type=None, null_marker=None, parquet_map_target_type=None):
    """Creates an external table definition with the given URIs and the schema.

  Arguments:
    source_format: Format of source data. For CSV files, specify 'CSV'. For
      Google spreadsheet files, specify 'GOOGLE_SHEETS'. For newline-delimited
      JSON, specify 'NEWLINE_DELIMITED_JSON'. For Cloud Datastore backup,
      specify 'DATASTORE_BACKUP'. For Avro files, specify 'AVRO'. For Orc files,
      specify 'ORC'. For Parquet files, specify 'PARQUET'. For Iceberg tables,
      specify 'ICEBERG'.
    source_uris: Comma separated list of URIs that contain data for this table.
    schema: Either an inline schema or path to a schema file.
    autodetect: Indicates if format options, compression mode and schema be auto
      detected from the source data. True - means that autodetect is on, False
      means that it is off. None means format specific default: - For CSV it
      means autodetect is OFF - For JSON it means that autodetect is ON. For
      JSON, defaulting to autodetection is safer because the only option
      autodetected is compression. If a schema is passed, then the user-supplied
      schema is used.
    connection_id: The user flag with the same name defined for the _Load
      BigqueryCmd
    ignore_unknown_values:  Indicates if BigQuery should allow extra values that
      are not represented in the table schema. If true, the extra values are
      ignored. If false, records with extra columns are treated as bad records,
      and if there are too many bad records, an invalid error is returned in the
      job result. The default value is false. The sourceFormat property
      determines what BigQuery treats as an extra value: - CSV: Trailing columns
      - JSON: Named values that don't match any column names.
    hive_partitioning_mode: Enables hive partitioning.  AUTO indicates to
      perform automatic type inference.  STRINGS indicates to treat all hive
      partition keys as STRING typed.  No other values are accepted.
    hive_partitioning_source_uri_prefix: Shared prefix for all files until hive
      partitioning encoding begins.
    require_hive_partition_filter: The user flag with the same name defined for
      the _Load BigqueryCmd
    use_avro_logical_types: The user flag with the same name defined for the
      _Load BigqueryCmd
    parquet_enum_as_string: The user flag with the same name defined for the
      _Load BigqueryCmd
    parquet_enable_list_inference: The user flag with the same name defined for
      the _Load BigqueryCmd
    metadata_cache_mode: Enables metadata cache for an external table with a
      connection. Specify 'AUTOMATIC' to automatically refresh the cached
      metadata. Specify 'MANUAL' to stop the automatic refresh.
    object_metadata: Object Metadata Type.
    preserve_ascii_control_characters: The user flag with the same name defined
      for the _Load BigqueryCmd
    reference_file_schema_uri: The user flag with the same name defined for the
      _Load BigqueryCmd
    encoding: Encoding types for CSV files. Available options are: 'UTF-8',
      'ISO-8859-1', 'UTF-16BE', 'UTF-16LE', 'UTF-32BE', and 'UTF-32LE'. The
      default value is 'UTF-8'.
    file_set_spec_type: Set how to discover files given source URIs. Specify
      'FILE_SYSTEM_MATCH' (default behavior) to expand source URIs by listing
      files from the underlying object store. Specify
      'NEW_LINE_DELIMITED_MANIFEST' to parse the URIs as new line delimited
      manifest files, where each line contains a URI (No wild-card URIs are
      supported).
    null_marker: Specifies a string that represents a null value in a CSV file.
    parquet_map_target_type: Indicate the target type for parquet maps. If
      unspecified, we represent parquet maps as map {repeated key_value {key,
      value}}. This option can simplify this by omiting the key_value record if
      it's equal to ARRAY_OF_STRUCT.

  Returns:
    A python dictionary that contains a external table definition for the given
    format with the most common options set.
  """
    try:
        supported_formats = ['CSV', 'NEWLINE_DELIMITED_JSON', 'DATASTORE_BACKUP', 'DELTA_LAKE', 'AVRO', 'ORC', 'PARQUET', 'GOOGLE_SHEETS', 'ICEBERG']
        if source_format not in supported_formats:
            raise app.UsageError('%s is not a supported format.' % source_format)
        external_table_def = {'sourceFormat': source_format}
        if file_set_spec_type is not None:
            external_table_def['fileSetSpecType'] = file_set_spec_type
        if metadata_cache_mode is not None:
            external_table_def['metadataCacheMode'] = metadata_cache_mode
        if object_metadata is not None:
            supported_obj_metadata_types = ['DIRECTORY', 'SIMPLE']
            if object_metadata not in supported_obj_metadata_types:
                raise app.UsageError('%s is not a supported Object Metadata Type.' % object_metadata)
            external_table_def['sourceFormat'] = None
            external_table_def['objectMetadata'] = object_metadata
        if external_table_def['sourceFormat'] == 'CSV':
            if autodetect:
                external_table_def['autodetect'] = True
                external_table_def['csvOptions'] = yaml.safe_load('\n            {\n                "quote": \'"\',\n                "encoding": "UTF-8"\n            }\n        ')
            else:
                external_table_def['csvOptions'] = yaml.safe_load('\n            {\n                "allowJaggedRows": false,\n                "fieldDelimiter": ",",\n                "allowQuotedNewlines": false,\n                "quote": \'"\',\n                "skipLeadingRows": 0,\n                "encoding": "UTF-8"\n            }\n        ')
            external_table_def['csvOptions']['preserveAsciiControlCharacters'] = preserve_ascii_control_characters
            external_table_def['csvOptions']['encoding'] = encoding or 'UTF-8'
            if null_marker is not None:
                external_table_def['csvOptions']['nullMarker'] = null_marker
        elif external_table_def['sourceFormat'] == 'NEWLINE_DELIMITED_JSON':
            if autodetect is None or autodetect:
                external_table_def['autodetect'] = True
            external_table_def['jsonOptions'] = {'encoding': encoding or 'UTF-8'}
        elif external_table_def['sourceFormat'] == 'GOOGLE_SHEETS':
            if autodetect is None or autodetect:
                external_table_def['autodetect'] = True
            else:
                external_table_def['googleSheetsOptions'] = yaml.safe_load('\n            {\n                "skipLeadingRows": 0\n            }\n        ')
        elif external_table_def['sourceFormat'] == 'AVRO':
            external_table_def['avroOptions'] = {'useAvroLogicalTypes': use_avro_logical_types}
            if reference_file_schema_uri is not None:
                external_table_def['referenceFileSchemaUri'] = reference_file_schema_uri
        elif external_table_def['sourceFormat'] == 'PARQUET':
            external_table_def['parquetOptions'] = {'enumAsString': parquet_enum_as_string, 'enableListInference': parquet_enable_list_inference, 'mapTargetType': parquet_map_target_type}
            if reference_file_schema_uri is not None:
                external_table_def['referenceFileSchemaUri'] = reference_file_schema_uri
        elif external_table_def['sourceFormat'] == 'ORC':
            if reference_file_schema_uri is not None:
                external_table_def['referenceFileSchemaUri'] = reference_file_schema_uri
        elif external_table_def['sourceFormat'] == 'ICEBERG' or external_table_def['sourceFormat'] == 'DELTA_LAKE':
            source_format = 'Iceberg' if external_table_def['sourceFormat'] == 'ICEBERG' else 'Delta Lake'
            if autodetect is not None and (not autodetect) or schema:
                raise app.UsageError('Cannot create %s table from user-specified schema.' % (source_format,))
            external_table_def['autodetect'] = True
            if len(source_uris.split(',')) != 1:
                raise app.UsageError('Must provide only one source_uri for %s table.' % (source_format,))
        if ignore_unknown_values:
            external_table_def['ignoreUnknownValues'] = True
        if hive_partitioning_mode is not None:
            ValidateHivePartitioningOptions(hive_partitioning_mode)
            hive_partitioning_options = {}
            hive_partitioning_options['mode'] = hive_partitioning_mode
            if hive_partitioning_source_uri_prefix is not None:
                hive_partitioning_options['sourceUriPrefix'] = hive_partitioning_source_uri_prefix
            external_table_def['hivePartitioningOptions'] = hive_partitioning_options
            if require_hive_partition_filter:
                hive_partitioning_options['requirePartitionFilter'] = True
        if schema:
            fields = bq_client_utils.ReadSchema(schema)
            external_table_def['schema'] = {'fields': fields}
        if connection_id:
            external_table_def['connectionId'] = connection_id
        external_table_def['sourceUris'] = source_uris.split(',')
        return external_table_def
    except ValueError as e:
        raise app.UsageError('Error occurred while creating table definition: %s' % e)