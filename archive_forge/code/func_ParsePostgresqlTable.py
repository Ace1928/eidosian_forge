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
def ParsePostgresqlTable(messages, postgresql_table_object):
    """Parses a raw postgresql table json/yaml into the PostgresqlTable message."""
    postgresql_columns_msg_list = []
    for column in postgresql_table_object.get('postgresql_columns', []):
        postgresql_columns_msg_list.append(ParsePostgresqlColumn(messages, column))
    table_name = postgresql_table_object.get('table')
    if not table_name:
        raise ds_exceptions.ParseError('Cannot parse YAML: missing key "table".')
    return messages.PostgresqlTable(table=table_name, postgresqlColumns=postgresql_columns_msg_list)