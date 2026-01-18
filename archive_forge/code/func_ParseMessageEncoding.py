from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import exceptions
def ParseMessageEncoding(messages, message_encoding):
    enc = message_encoding.lower()
    if enc == 'json':
        return messages.SchemaSettings.EncodingValueValuesEnum.JSON
    elif enc == 'binary':
        return messages.SchemaSettings.EncodingValueValuesEnum.BINARY
    else:
        raise InvalidSchemaSettingsException('Unknown message encoding. Options are JSON or BINARY.')