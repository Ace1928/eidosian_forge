from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
import json
import os
import subprocess
import sys
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import semver
import six
from six.moves import urllib
def GetNormalizedURL(server):
    """Sanitize and normalize the server input."""
    parsed_url = urllib.parse.urlparse(server)
    if '://' not in server:
        parsed_url = urllib.parse.urlparse('https://' + server)
        if parsed_url.hostname == 'localhost':
            parsed_url = urllib.parse.urlparse('http://' + server)
    return parsed_url