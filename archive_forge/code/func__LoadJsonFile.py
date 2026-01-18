from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.auth import service_account
from googlecloudsdk.command_lib.artifacts import requests as ar_requests
from googlecloudsdk.command_lib.artifacts import util as ar_util
from googlecloudsdk.command_lib.artifacts.print_settings import apt
from googlecloudsdk.command_lib.artifacts.print_settings import gradle
from googlecloudsdk.command_lib.artifacts.print_settings import mvn
from googlecloudsdk.command_lib.artifacts.print_settings import npm
from googlecloudsdk.command_lib.artifacts.print_settings import python
from googlecloudsdk.command_lib.artifacts.print_settings import yum
from googlecloudsdk.core import config
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import creds
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.credentials import store
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
def _LoadJsonFile(filename):
    """Checks and validates if given filename is a proper JSON file.

  Args:
    filename: str, path to JSON file.

  Returns:
    bytes, the content of the file.
  """
    content = console_io.ReadFromFileOrStdin(filename, binary=True)
    try:
        json.loads(encoding.Decode(content))
        return content
    except ValueError as e:
        if filename.endswith('.json'):
            raise service_account.BadCredentialFileException('Could not read JSON file {0}: {1}'.format(filename, e))
    raise service_account.BadCredentialFileException('Unsupported credential file: {0}'.format(filename))