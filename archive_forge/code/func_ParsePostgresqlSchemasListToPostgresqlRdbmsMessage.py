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
def ParsePostgresqlSchemasListToPostgresqlRdbmsMessage(messages, postgresql_rdbms_data):
    """Parses an object of type {postgresql_schemas: [...]} into the PostgresqlRdbms message."""
    postgresql_schemas_raw = postgresql_rdbms_data.get('postgresql_schemas', [])
    postgresql_schema_msg_list = []
    for schema in postgresql_schemas_raw:
        postgresql_schema_msg_list.append(ParsePostgresqlSchema(messages, schema))
    postgresql_rdbms_msg = messages.PostgresqlRdbms(postgresqlSchemas=postgresql_schema_msg_list)
    return postgresql_rdbms_msg