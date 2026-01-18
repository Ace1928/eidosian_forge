from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
import stat
import tarfile
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import local_file_adapter
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import http_encoding
from googlecloudsdk.core.util import retry
import requests
import six
def _ExecuteRequestAndRaiseExceptions(url, headers, timeout):
    """Executes an HTTP request using requests.

  Args:
    url: str, the url to download.
    headers: obj, the headers to include in the request.
    timeout: int, the timeout length for the request.

  Returns:
    A response object from the request.

  Raises:
    requests.exceptions.HTTPError in the case of a client or server error.
  """
    from googlecloudsdk.core import requests as core_requests
    requests_session = core_requests.GetSession()
    if url.startswith('file://'):
        requests_session.mount('file://', local_file_adapter.LocalFileAdapter())
    response = requests_session.get(url, headers=headers, timeout=timeout, stream=True)
    response.raise_for_status()
    return response