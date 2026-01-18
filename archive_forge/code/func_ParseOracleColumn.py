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
def ParseOracleColumn(messages, oracle_column_object, release_track):
    """Parses a raw oracle column json/yaml into the OracleColumn message."""
    message = messages.OracleColumn(column=oracle_column_object.get(_GetRDBMSFieldName('column', release_track), ''))
    data_type = oracle_column_object.get('data_type')
    if data_type is not None:
        message.dataType = data_type
    encoding = oracle_column_object.get('encoding')
    if encoding is not None:
        message.encoding = encoding
    length = oracle_column_object.get('length')
    if length is not None:
        message.length = length
    nullable = oracle_column_object.get('nullable')
    if nullable is not None:
        message.nullable = nullable
    ordinal_position = oracle_column_object.get('ordinal_position')
    if ordinal_position is not None:
        message.ordinalPosition = ordinal_position
    precision = oracle_column_object.get('precision')
    if precision is not None:
        message.precision = precision
    primary_key = oracle_column_object.get('primary_key')
    if primary_key is not None:
        message.primaryKey = primary_key
    scale = oracle_column_object.get('scale')
    if scale is not None:
        message.scale = scale
    return message