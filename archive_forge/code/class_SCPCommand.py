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
class SCPCommand(object):
    """Represents a platform independent SCP command.

  This class is intended to manage the most important suite- and platform
  specifics. We manage the following data:
  - The executable to call, either `scp` or `pscp`.
  - User and host, through either `sources` or `destination` arg. Multiple
    remote sources are allowed but not supported under PuTTY. Multiple local
    sources are always allowed.
  - Potential remote command to execute, `remote_command` arg.

  In addition, it manages these flags:
  -r          Recursive copy
  -C          Compression
  -P          Port
  -i          Identity file (private key)
  -o Key=Val  OpenSSH specific options that should be added, `options` arg.

  For flexibility, SCPCommand also accepts `extra_flags`. Always use these
  with caution -- they will be added as-is to the command invocation without
  validation. Specifically, do not add any of the above mentioned flags.
  """

    def __init__(self, sources, destination, recursive=False, compress=False, port=None, identity_file=None, options=None, extra_flags=None, iap_tunnel_args=None, identity_list=None):
        """Construct a suite independent SCP command.

    Args:
      sources: [FileReference] or FileReference, the source(s) for this copy. At
        least one source is required. NOTE: Multiple remote sources are not
        supported in PuTTY and is discouraged for consistency.
      destination: FileReference, the destination file or directory. If remote
        source, this must be local, and vice versa.
      recursive: bool, recursive directory copy.
      compress: bool, enable compression.
      port: str, port.
      identity_file: str, path to private key file.
      options: {str: str}, options (`-o`) for OpenSSH, see `ssh_config(5)`.
      extra_flags: [str], extra flags to append to scp invocation. Both binary
        style flags `['-b']` and flags with values `['-k', 'v']` are accepted.
      iap_tunnel_args: iap_tunnel.SshTunnelArgs or None, options about IAP
        Tunnel.
      identity_list: list, A list of paths to private key files. Overrides the
        identity_file argument, and sets multiple `['-i']` flags.
    """
        self.sources = [sources] if isinstance(sources, FileReference) else sources
        self.destination = destination
        self.recursive = recursive
        self.compress = compress
        self.port = port
        self.identity_file = identity_file
        self.identity_list = identity_list
        self.options = options or {}
        self.extra_flags = extra_flags or []
        self.iap_tunnel_args = iap_tunnel_args

    @classmethod
    def Verify(cls, sources, destination, single_remote=False, env=None):
        """Verify that the source- and destination config is sound.

    Checks that sources are remote if destination is local and vice versa,
    plus raises error for multiple remote sources in PuTTY, which is not
    supported by `pscp`.

    Args:
      sources: [FileReference], see SCPCommand.sources.
      destination: FileReference, see SCPCommand.destination.
      single_remote: bool, if True, enforce that all remote sources refer to the
        same Remote (user and host).
      env: Environment, the current environment.

    Raises:
      InvalidConfigurationError: The source/destination configuration is
        invalid.
    """
        env = env or Environment.Current()
        if not sources:
            raise InvalidConfigurationError('No sources provided.', sources, destination)
        if destination.remote:
            if any([src.remote for src in sources]):
                raise InvalidConfigurationError('All sources must be local files when destination is remote.', sources, destination)
        else:
            if env.suite is Suite.PUTTY and len(sources) != 1:
                raise InvalidConfigurationError('Multiple remote sources not supported by PuTTY.', sources, destination)
            if not all([src.remote for src in sources]):
                raise InvalidConfigurationError('Source(s) must be remote when destination is local.', sources, destination)
            if single_remote and len(set([src.remote for src in sources])) != 1:
                raise InvalidConfigurationError('All sources must refer to the same remote when destination is local.', sources, destination)

    def Build(self, env=None):
        """Construct the actual command according to the given environment.

    Args:
      env: Environment, to construct the command for (or current if None).

    Raises:
      InvalidConfigurationError: The source/destination configuration is
        invalid.
      MissingCommandError: If SCP command(s) required were not found.

    Returns:
      [str], the command args (where the first arg is the command itself).
    """
        env = env or Environment.Current()
        if not env.scp:
            raise MissingCommandError('The current environment lacks an SCP (secure copy) client.')
        self.Verify(self.sources, self.destination, env=env)
        args = [env.scp]
        if self.recursive:
            args.append('-r')
        if self.compress:
            args.append('-C')
        if self.port:
            args.extend(['-P', self.port])
        if self.identity_list:
            for identity_file in self.identity_list:
                args.extend(['-i', identity_file])
        elif self.identity_file:
            identity_file = self.identity_file
            if env.suite is Suite.PUTTY and (not identity_file.endswith('.ppk')):
                identity_file += '.ppk'
            args.extend(['-i', identity_file])
        if env.suite is Suite.OPENSSH:
            for key, value in sorted(six.iteritems(self.options)):
                args.extend(['-o', '{k}={v}'.format(k=key, v=value)])
        args.extend(_BuildIapTunnelProxyCommandArgs(self.iap_tunnel_args, env))
        args.extend(self.extra_flags)
        args.extend([source.ToArg() for source in self.sources])
        args.append(self.destination.ToArg())
        return args

    def Run(self, env=None, putty_force_connect=False):
        """Run the SCP command using the given environment.

    Args:
      env: Environment, environment to run in (or current if None).
      putty_force_connect: bool, whether to inject 'y' into the prompts for
        `pscp`, which is insecure and not recommended. It serves legacy
        compatibility purposes for existing usages only; DO NOT SET THIS IN NEW
        CODE.

    Raises:
      InvalidConfigurationError: The source/destination configuration is
        invalid.
      MissingCommandError: If SCP command(s) not found.
      CommandError: SCP command failed to copy the file(s).
    """
        env = env or Environment.Current()
        args = self.Build(env)
        log.debug('Running command [{}].'.format(' '.join(args)))
        if env.suite is Suite.PUTTY and putty_force_connect:
            in_str = 'y\n'
        else:
            in_str = None
        status = execution_utils.Exec(args, no_exit=True, in_str=in_str)
        if status:
            raise CommandError(args[0], return_code=status)