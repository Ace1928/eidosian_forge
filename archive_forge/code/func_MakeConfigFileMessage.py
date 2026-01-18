from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.endpoints import config_reporter
from googlecloudsdk.api_lib.endpoints import exceptions
from googlecloudsdk.api_lib.endpoints import services_util
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import exceptions as services_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import http_encoding
import six.moves.urllib.parse
def MakeConfigFileMessage(self, file_contents, filename, file_type):
    """Constructs a ConfigFile message from a config file.

    Args:
      file_contents: The contents of the config file.
      filename: The full path to the config file.
      file_type: FileTypeValueValuesEnum describing the type of config file.

    Returns:
      The constructed ConfigFile message.
    """
    messages = services_util.GetMessagesModule()
    file_types = messages.ConfigFile.FileTypeValueValuesEnum
    if file_type != file_types.FILE_DESCRIPTOR_SET_PROTO:
        file_contents = http_encoding.Encode(file_contents)
    return messages.ConfigFile(fileContents=file_contents, filePath=os.path.basename(filename), fileType=file_type)