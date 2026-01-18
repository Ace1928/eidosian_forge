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
def CreateSSHPoller(remote, identity_file, options, iap_tunnel_args, extra_flags=None, port=None):
    """Creates and returns an SSH poller."""
    ssh_poller_args = {'remote': remote, 'identity_file': identity_file, 'options': options, 'iap_tunnel_args': iap_tunnel_args, 'extra_flags': extra_flags, 'max_wait_ms': SSH_KEY_PROPAGATION_TIMEOUT_MS}
    if port:
        ssh_poller_args['port'] = six.text_type(port)
    return ssh.SSHPoller(**ssh_poller_args)