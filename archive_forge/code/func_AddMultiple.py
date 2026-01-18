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
def AddMultiple(self, hostname, host_keys, overwrite=False):
    """Add or update multiple entries for the given hostname.

    If there is no entry for the given hostname, the keys will be added. If
    there is an entry already, and overwrite keys is False, nothing will be
    changed. If there is an entry and overwrite_keys is True, all  current
    entries for the given hostname will be removed and the new keys added.

    Args:
      hostname: str, The hostname for the known_hosts entry.
      host_keys: list, A list of host keys for the given hostname.
      overwrite: bool, If true, will overwrite the entries corresponding to
        hostname with the new host_key if it already exists. If false and an
        entry already exists for hostname, will ignore the new host_key values.

    Returns:
      bool, True if new keys were added.
    """
    new_keys_added = False
    new_key_entries = ['{0} {1}'.format(hostname, host_key) for host_key in host_keys]
    if not new_key_entries:
        return new_keys_added
    existing_entries = [key for key in self.known_hosts if key.startswith(hostname)]
    if existing_entries:
        if overwrite:
            self.known_hosts = [key for key in self.known_hosts if not key.startswith(hostname)]
            self.known_hosts.extend(new_key_entries)
            new_keys_added = True
    else:
        self.known_hosts.extend(new_key_entries)
        new_keys_added = True
    return new_keys_added