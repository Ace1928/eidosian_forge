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
def MetadataHasEnable(metadata, key_name):
    """Return true if the metadata has the supplied key and it is set to 'true'.

  Args:
    metadata: Instance or Project metadata.
    key_name: The name of the metadata key to check. e.g. 'oslogin-enable'.

  Returns:
    True if Enabled, False if Disabled, None if key is not presesnt.
  """
    if not (metadata and metadata.items):
        return None
    matching_values = [item.value for item in metadata.items if item.key == key_name]
    if not matching_values:
        return None
    return matching_values[0].lower() == 'true'