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
class KnownHosts(object):
    """Represents known hosts file, supports read, write and basic key management.

  Currently a very naive, but sufficient, implementation where each entry is
  simply a string, and all entries are list of those strings.
  """
    DEFAULT_PATH = os.path.realpath(files.ExpandHomeDir(os.path.join('~', '.ssh', 'google_compute_known_hosts')))

    def __init__(self, known_hosts, file_path):
        """Construct a known hosts representation based on a list of key strings.

    Args:
      known_hosts: str, list each corresponding to a line in known_hosts_file.
      file_path: str, path to the known_hosts_file.
    """
        self.known_hosts = known_hosts
        self.file_path = file_path

    @classmethod
    def FromFile(cls, file_path):
        """Create a KnownHosts object given a known_hosts_file.

    Args:
      file_path: str, path to the known_hosts_file.

    Returns:
      KnownHosts object corresponding to the file. If the file could not be
      opened, the KnownHosts object will have no entries.
    """
        try:
            known_hosts = files.ReadFileContents(file_path).splitlines()
        except files.Error as e:
            known_hosts = []
            log.debug('SSH Known Hosts File [{0}] could not be opened: {1}'.format(file_path, e))
        return KnownHosts(known_hosts, file_path)

    @classmethod
    def FromDefaultFile(cls):
        """Create a KnownHosts object from the default known_hosts_file.

    Returns:
      KnownHosts object corresponding to the default known_hosts_file.
    """
        return KnownHosts.FromFile(KnownHosts.DEFAULT_PATH)

    def ContainsAlias(self, host_key_alias):
        """Check if a host key alias exists in one of the known hosts.

    Args:
      host_key_alias: str, the host key alias

    Returns:
      bool, True if host_key_alias is in the known hosts file. If the known
      hosts file couldn't be opened it will be treated as if empty and False
      returned.
    """
        return any((host_key_alias in line for line in self.known_hosts))

    def Add(self, hostname, host_key, overwrite=False):
        """Add or update the entry for the given hostname.

    If there is no entry for the given hostname, it will be added. If there is
    an entry already and overwrite_keys is False, nothing will be changed. If
    there is an entry and overwrite_keys is True, the key will be updated if it
    has changed.

    Args:
      hostname: str, The hostname for the known_hosts entry.
      host_key: str, The host key for the given hostname.
      overwrite: bool, If true, will overwrite the entry corresponding to
        hostname with the new host_key if it already exists. If false and an
        entry already exists for hostname, will ignore the new host_key value.
    """
        new_key_entry = '{0} {1}'.format(hostname, host_key)
        for i, key in enumerate(self.known_hosts):
            if key.startswith(hostname):
                if overwrite:
                    self.known_hosts[i] = new_key_entry
                break
        else:
            self.known_hosts.append(new_key_entry)

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

    def Write(self):
        """Writes the file to disk."""
        files.WriteFileContents(self.file_path, '\n'.join(self.known_hosts) + '\n', private=True)