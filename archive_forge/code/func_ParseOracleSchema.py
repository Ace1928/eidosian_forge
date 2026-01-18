from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import uuid
from apitools.base.py import encoding as api_encoding
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.datastream import camel_case_utils
from googlecloudsdk.api_lib.datastream import exceptions as ds_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
import six
def ParseOracleSchema(messages, oracle_schema_object, release_track):
    """Parses a raw oracle schema json/yaml into the OracleSchema message."""
    oracle_tables_msg_list = []
    for table in oracle_schema_object.get('oracle_tables', []):
        oracle_tables_msg_list.append(ParseOracleTable(messages, table, release_track))
    schema_key = _GetRDBMSFieldName('schema', release_track)
    schema_name = oracle_schema_object.get(schema_key)
    if not schema_name:
        raise ds_exceptions.ParseError('Cannot parse YAML: missing key "%s".' % schema_key)
    return messages.OracleSchema(schema=schema_name, oracleTables=oracle_tables_msg_list)