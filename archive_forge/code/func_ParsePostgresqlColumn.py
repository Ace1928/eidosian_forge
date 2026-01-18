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
def ParsePostgresqlColumn(messages, postgresql_column_object):
    """Parses a raw postgresql column json/yaml into the PostgresqlColumn message."""
    message = messages.PostgresqlColumn(column=postgresql_column_object.get('column', ''))
    data_type = postgresql_column_object.get('data_type')
    if data_type is not None:
        message.dataType = data_type
    length = postgresql_column_object.get('length')
    if length is not None:
        message.length = length
    precision = postgresql_column_object.get('precision')
    if precision is not None:
        message.precision = precision
    scale = postgresql_column_object.get('scale')
    if scale is not None:
        message.scale = scale
    primary_key = postgresql_column_object.get('primary_key')
    if primary_key is not None:
        message.primaryKey = primary_key
    nullable = postgresql_column_object.get('nullable')
    if nullable is not None:
        message.nullable = nullable
    ordinal_position = postgresql_column_object.get('ordinal_position')
    if ordinal_position is not None:
        message.ordinalPosition = ordinal_position
    return message