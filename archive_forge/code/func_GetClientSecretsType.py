from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import json
import textwrap
from googlecloudsdk.command_lib.util import check_browser
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
import six
def GetClientSecretsType(client_id_file):
    """Get the type of the client secrets file (web or installed)."""
    invalid_file_format_msg = 'Invalid file format. See https://developers.google.com/api-client-library/python/guide/aaa_client_secrets'
    try:
        obj = json.loads(files.ReadFileContents(client_id_file))
    except files.Error:
        raise InvalidClientSecretsError('Cannot read file: "%s"' % client_id_file)
    if obj is None:
        raise InvalidClientSecretsError(invalid_file_format_msg)
    if len(obj) != 1:
        raise InvalidClientSecretsError(invalid_file_format_msg + ' Expected a JSON object with a single property for a "web" or "installed" application')
    return tuple(obj)[0]