from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import enum
import errno
import getpass
import os
import re
import string
import subprocess
import tempfile
import textwrap
from googlecloudsdk.api_lib.oslogin import client as oslogin_client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.oslogin import oslogin_utils
from googlecloudsdk.command_lib.util import gaia
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import retry
import six
from six.moves.urllib.parse import quote
def _StatusMessage(self):
    """Prepares human readable SSH key status information."""
    messages = []
    key_padding = 0
    status_padding = 0
    for kind in self.keys:
        data = self.keys[kind]
        key_padding = max(key_padding, len(kind.value))
        status_padding = max(status_padding, len(data.status.value))
    for kind in self.keys:
        data = self.keys[kind]
        messages.append('{} {} [{}]\n'.format((kind.value + ' key').ljust(key_padding + 4), ('(' + data.status.value + ')').ljust(status_padding + 2), data.filename))
    messages.sort()
    return ''.join(messages)