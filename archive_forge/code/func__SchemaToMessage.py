from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from os import path
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from googlecloudsdk.api_lib.data_catalog import entries_v1
from googlecloudsdk.api_lib.data_catalog import util as api_util
from googlecloudsdk.command_lib.concepts import exceptions as concept_exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def _SchemaToMessage(schema):
    """Converts the given schema dict to the corresponding schema message.

  Args:
    schema: dict, The schema that has been processed.

  Returns:
    googleCloudDatacatalogV1betaSchema
  Raises:
    InvalidSchemaError: If the schema is invalid.
  """
    messages = api_util.GetMessagesModule('v1')
    try:
        schema_message = encoding.DictToMessage({'columns': schema}, messages.GoogleCloudDatacatalogV1Schema)
    except AttributeError:
        raise InvalidSchemaError('Invalid schema: expected list of column names along with their types, modes, descriptions, and/or nested subcolumns.')
    except _messages.ValidationError as e:
        raise InvalidSchemaError('Invalid schema: [{}]'.format(e))
    unrecognized_field_paths = _GetUnrecognizedFieldPaths(schema_message)
    if unrecognized_field_paths:
        error_msg_lines = ['Invalid schema, the following fields are unrecognized:']
        error_msg_lines += unrecognized_field_paths
        raise InvalidSchemaError('\n'.join(error_msg_lines))
    return schema_message