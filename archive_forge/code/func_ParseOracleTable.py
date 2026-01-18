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
def ParseOracleTable(messages, oracle_table_object, release_track):
    """Parses a raw oracle table json/yaml into the OracleTable message."""
    oracle_columns_msg_list = []
    for column in oracle_table_object.get('oracle_columns', []):
        oracle_columns_msg_list.append(ParseOracleColumn(messages, column, release_track))
    table_key = _GetRDBMSFieldName('table', release_track)
    table_name = oracle_table_object.get(table_key)
    if not table_name:
        raise ds_exceptions.ParseError('Cannot parse YAML: missing key "%s".' % table_key)
    return messages.OracleTable(table=table_name, oracleColumns=oracle_columns_msg_list)