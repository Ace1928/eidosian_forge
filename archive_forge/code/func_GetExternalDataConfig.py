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
def GetExternalDataConfig(file_path_or_simple_spec, use_avro_logical_types=False, parquet_enum_as_string=False, parquet_enable_list_inference=False, metadata_cache_mode=None, object_metadata=None, preserve_ascii_control_characters=None, reference_file_schema_uri=None, file_set_spec_type=None, null_marker=None, parquet_map_target_type=None):
    """Returns a ExternalDataConfiguration from the file or specification string.

  Determines if the input string is a file path or a string,
  then returns either the parsed file contents, or the parsed configuration from
  string. The file content is expected to be JSON representation of
  ExternalDataConfiguration. The specification is expected to be of the form
  schema@format=uri i.e. schema is separated from format and uri by '@'. If the
  uri itself contains '@' or '=' then the JSON file option should be used.
  "format=" can be omitted for CSV files.

  Raises:
    UsageError: when incorrect usage or invalid args are used.
  """
    maybe_filepath = os.path.expanduser(file_path_or_simple_spec)
    if os.path.isfile(maybe_filepath):
        try:
            with open(maybe_filepath) as external_config_file:
                return yaml.safe_load(external_config_file)
        except yaml.error.YAMLError as e:
            raise app.UsageError('Error decoding YAML external table definition from file %s: %s' % (maybe_filepath, e))
    else:
        source_format = 'CSV'
        schema = None
        connection_id = None
        error_msg = 'Error decoding external_table_definition. external_table_definition should either be the name of a JSON file or the text representation of an external table definition. Given:%s' % file_path_or_simple_spec
        parts = file_path_or_simple_spec.split('@')
        if len(parts) == 1:
            format_and_uri = parts[0]
        elif len(parts) == 2:
            if parts[0].find('://') >= 0:
                format_and_uri = parts[0]
                connection_id = parts[1]
            else:
                schema = parts[0]
                format_and_uri = parts[1]
        elif len(parts) == 3:
            schema = parts[0]
            format_and_uri = parts[1]
            connection_id = parts[2]
        else:
            raise app.UsageError(error_msg)
        separator_pos = format_and_uri.find('=')
        if separator_pos < 0:
            uri = format_and_uri
        else:
            source_format = format_and_uri[0:separator_pos]
            uri = format_and_uri[separator_pos + 1:]
        if not uri:
            raise app.UsageError(error_msg)
        return CreateExternalTableDefinition(source_format, uri, schema, True, connection_id, use_avro_logical_types=use_avro_logical_types, parquet_enum_as_string=parquet_enum_as_string, parquet_enable_list_inference=parquet_enable_list_inference, metadata_cache_mode=metadata_cache_mode, object_metadata=object_metadata, preserve_ascii_control_characters=preserve_ascii_control_characters, reference_file_schema_uri=reference_file_schema_uri, file_set_spec_type=file_set_spec_type, null_marker=null_marker, parquet_map_target_type=parquet_map_target_type)