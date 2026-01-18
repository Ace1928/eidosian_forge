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
def IsCertValid(cert):
    time_format = '%Y-%m-%dT%H:%M:%S'
    match = re.findall('\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}', cert)
    if not match:
        return
    start = datetime.datetime.strptime(match[0], time_format)
    end = datetime.datetime.strptime(match[1], time_format)
    now = datetime.datetime.now()
    oslogin_state.signed_ssh_key = now > start and now < end