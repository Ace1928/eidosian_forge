from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import re
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai.custom_jobs import local_util
from googlecloudsdk.core import log
def _ParseHostPort(host):
    """Parses a registry hostname to a list of components and an optional port.

  Args:
    host: (str) The registry hostname supposed to comply with standard DNS
      rules, optionally be followed by a port number in the format like ":8080".

  Returns:
    A (hostcomponents, port) tuple representing the parsed result.
    The hostcomponents contains each dot-seperated component in the given
    hostname; port may be None if it isn't present.
  """
    parts = host.rsplit(':', 1)
    hostcomponents = parts[0].split('.')
    port = parts[1] if len(parts) == 2 else None
    return (hostcomponents, port)