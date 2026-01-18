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
def ParseOracleSchemasListToOracleRdbmsMessage(messages, oracle_rdbms_data, release_track=base.ReleaseTrack.BETA):
    """Parses an object of type {oracle_schemas: [...]} into the OracleRdbms message."""
    oracle_schemas_raw = oracle_rdbms_data.get('oracle_schemas', [])
    oracle_schema_msg_list = []
    for schema in oracle_schemas_raw:
        oracle_schema_msg_list.append(ParseOracleSchema(messages, schema, release_track))
    oracle_rdbms_msg = messages.OracleRdbms(oracleSchemas=oracle_schema_msg_list)
    return oracle_rdbms_msg