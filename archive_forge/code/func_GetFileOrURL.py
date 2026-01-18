from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import copy
import json
import os
from googlecloudsdk.command_lib.anthos import flags
from googlecloudsdk.command_lib.anthos.common import file_parsers
from googlecloudsdk.command_lib.anthos.common import messages
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import exceptions as c_except
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import requests
import six
from six.moves import urllib
def GetFileOrURL(cluster_config, certificate_file=True):
    """Parses config input to determine whether URL or File logic should execute.

     Determines whether the cluster_config is a file or URL. If it's a URL, it
     then pulls the contents of the file using a GET request. If it's a
     file, then it expands the file path and returns its contents.

  Args:
    cluster_config: str, A file path or URL for the login-config.
    certificate_file: str, Optional file path to the CA certificate to use with
      the GET request to the URL.

  Raises:
    AnthosAuthException: If the data could not be pulled from the URL.

  Returns:
    parsed_config_fileOrURL, config_contents, and is_url
    parsed_config_fileOrURL: str, returns either the URL that was passed or an
      expanded file path if a file was passed.
      config_contents: str, returns the contents of the file or URL.
    is_url: bool, True if the provided cluster_config input was a URL.
  """
    if not cluster_config:
        return (None, None, None)
    config_url = urllib.parse.urlparse(cluster_config)
    is_url = config_url.scheme == 'http' or config_url.scheme == 'https'
    if is_url:
        response = requests.get(cluster_config, verify=certificate_file or True)
        if response.status_code != requests.codes.ok:
            raise AnthosAuthException('Request to login-config URL failed withresponse code [{}] and text [{}]: '.format(response.status_code, response.text))
        return (cluster_config, response.text, is_url)
    expanded_config_path = flags.ExpandLocalDirAndVersion(cluster_config)
    contents = files.ReadFileContents(expanded_config_path)
    return (expanded_config_path, contents, is_url)