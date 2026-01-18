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
def ParseMysqlDatabase(messages, mysql_database_object, release_track):
    """Parses a raw mysql database json/yaml into the MysqlDatabase message."""
    mysql_tables_msg_list = []
    for table in mysql_database_object.get('mysql_tables', []):
        mysql_tables_msg_list.append(ParseMysqlTable(messages, table, release_track))
    database_key = _GetRDBMSFieldName('database', release_track)
    database_name = mysql_database_object.get(database_key)
    if not database_name:
        raise ds_exceptions.ParseError('Cannot parse YAML: missing key "%s".' % database_key)
    return messages.MysqlDatabase(database=database_name, mysqlTables=mysql_tables_msg_list)