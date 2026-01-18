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
def _BuildIapTunnelProxyCommandArgs(iap_tunnel_args, env):
    """Calculate the ProxyCommand flags for IAP Tunnel if necessary.

  IAP Tunnel with ssh runs an second inner version of gcloud by passing a
  command to do so as a ProxyCommand argument to OpenSSH/Putty.

  Args:
    iap_tunnel_args: iap_tunnel.SshTunnelArgs or None, options about IAP Tunnel.
    env: Environment, data about the ssh client.

  Raises:
    BadCharacterError: If instance arg contains any invalid characters.

  Returns:
    [str], the additional arguments for OpenSSH or Putty.
  """
    if not iap_tunnel_args:
        return []
    allowed_non_alnum_chars = {'-', '_', '.'}
    for char in iap_tunnel_args.instance:
        if not char.isalnum() and char not in allowed_non_alnum_chars:
            raise BadCharacterError('Instance name/IP/hostname contains illegal characters.')
    gcloud_command = execution_utils.ArgsForGcloud()
    gcloud_command = [_EscapeProxyCommandArg(x, env) for x in gcloud_command]
    if iap_tunnel_args.track:
        gcloud_command.append(iap_tunnel_args.track)
    port_token, quotation = ('%port', '"') if env.suite is Suite.PUTTY else ('%p', "'")
    gcloud_command.extend(['compute', 'start-iap-tunnel', quotation + iap_tunnel_args.instance + quotation, quotation + port_token + quotation, '--listen-on-stdin', '--project=' + iap_tunnel_args.project])
    if iap_tunnel_args.zone:
        gcloud_command.append('--zone=' + iap_tunnel_args.zone)
    if iap_tunnel_args.region:
        gcloud_command.append('--region=' + iap_tunnel_args.region)
    if iap_tunnel_args.network:
        gcloud_command.append('--network=' + iap_tunnel_args.network)
    for arg in iap_tunnel_args.pass_through_args:
        gcloud_command.append(_EscapeProxyCommandArg(arg, env))
    verbosity = log.GetVerbosityName()
    if verbosity:
        gcloud_command.append('--verbosity=' + verbosity)
    if env.suite is Suite.PUTTY:
        return ['-proxycmd', ' '.join(gcloud_command)]
    else:
        return ['-o', ' '.join(['ProxyCommand'] + gcloud_command), '-o', 'ProxyUseFdpass=no']