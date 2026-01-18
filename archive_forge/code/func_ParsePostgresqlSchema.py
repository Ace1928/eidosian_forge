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
def ParsePostgresqlSchema(messages, postgresql_schema_object):
    """Parses a raw postgresql schema json/yaml into the PostgresqlSchema message."""
    postgresql_tables_msg_list = []
    for table in postgresql_schema_object.get('postgresql_tables', []):
        postgresql_tables_msg_list.append(ParsePostgresqlTable(messages, table))
    schema_name = postgresql_schema_object.get('schema')
    if not schema_name:
        raise ds_exceptions.ParseError('Cannot parse YAML: missing key "schema".')
    return messages.PostgresqlSchema(schema=schema_name, postgresqlTables=postgresql_tables_msg_list)