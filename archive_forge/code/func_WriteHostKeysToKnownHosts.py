from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import collections
import datetime
import json
import os
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import times
from googlecloudsdk.core.util.files import FileReader
from googlecloudsdk.core.util.files import FileWriter
import six
def WriteHostKeysToKnownHosts(self, known_hosts, host_keys, host_key_alias):
    """Writes host keys to known hosts file.

    Only writes keys to known hosts file if there are no existing keys for
    the host.

    Args:
      known_hosts: obj, known_hosts file object.
      host_keys: dict, dictionary of host keys.
      host_key_alias: str, alias for host key entries.
    """
    host_key_entries = []
    for key_type, key in host_keys.items():
        host_key_entry = '{0} {1}'.format(key_type, key)
        host_key_entries.append(host_key_entry)
    host_key_entries.sort()
    new_keys_added = known_hosts.AddMultiple(host_key_alias, host_key_entries, overwrite=False)
    if new_keys_added:
        log.status.Print('Writing {0} keys to {1}'.format(len(host_key_entries), known_hosts.file_path))
    if host_key_entries and (not new_keys_added):
        log.status.Print('Existing host keys found in {0}'.format(known_hosts.file_path))
    known_hosts.Write()