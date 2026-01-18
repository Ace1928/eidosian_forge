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
class SSHCommand(object):
    """Represents a platform independent SSH command.

  This class is intended to manage the most important suite- and platform
  specifics. We manage the following data:
  - The executable to call, either `ssh`, `putty` or `plink`.
  - User and host, through the `remote` arg.
  - Potential remote command to execute, `remote_command` arg.

  In addition, it manages these flags:
  -t, -T      Pseudo-terminal allocation
  -p, -P      Port
  -i          Identity file (private key)
  -o Key=Val  OpenSSH specific options that should be added, `options` arg.

  For flexibility, SSHCommand also accepts `extra_flags`. Always use these
  with caution -- they will be added as-is to the command invocation without
  validation. Specifically, do not add any of the above mentioned flags.
  """

    def __init__(self, remote, port=None, identity_file=None, cert_file=None, options=None, extra_flags=None, remote_command=None, tty=None, iap_tunnel_args=None, remainder=None, identity_list=None):
        """Construct a suite independent SSH command.

    Note that `extra_flags` and `remote_command` arguments are lists of strings:
    `remote_command=['echo', '-e', 'hello']` is different from
    `remote_command=['echo', '-e hello']` -- the former is likely desired.
    For the same reason, `extra_flags` should be passed like `['-k', 'v']`.

    Args:
      remote: Remote, the remote to connect to.
      port: str, port.
      identity_file: str, path to private key file.
      cert_file: str, path to certificate file.
      options: {str: str}, options (`-o`) for OpenSSH, see `ssh_config(5)`.
      extra_flags: [str], extra flags to append to ssh invocation. Both binary
        style flags `['-b']` and flags with values `['-k', 'v']` are accepted.
      remote_command: [str], command to run remotely.
      tty: bool, launch a terminal. If None, determine automatically based on
        presence of remote command.
      iap_tunnel_args: iap_tunnel.SshTunnelArgs or None, options about IAP
        Tunnel.
      remainder: [str], NOT RECOMMENDED. Arguments to be appended directly to
        the native tool invocation, after the `[user@]host` part but prior to
        the remote command. On PuTTY, this can only be a remote command. On
        OpenSSH, this can be flags followed by a remote command. Cannot be
        combined with `remote_command`. Use `extra_flags` and `remote_command`
        instead.
      identity_list: list, A list of paths to private key files. Overrides the
        identity_file argument, and sets multiple `['-i']` flags.
    """
        self.remote = remote
        self.port = port
        self.identity_file = identity_file
        self.cert_file = cert_file
        self.identity_list = identity_list
        self.options = options or {}
        self.extra_flags = extra_flags or []
        self.remote_command = remote_command or []
        self.tty = tty
        self.iap_tunnel_args = iap_tunnel_args
        self.remainder = remainder
        self._remote_command_file = None

    def Build(self, env=None):
        """Construct the actual command according to the given environment.

    Args:
      env: Environment, to construct the command for (or current if None).

    Raises:
      MissingCommandError: If SSH command(s) required were not found.

    Returns:
      [str], the command args (where the first arg is the command itself).
    """
        env = env or Environment.Current()
        if not (env.ssh and env.ssh_term):
            raise MissingCommandError('The current environment lacks SSH.')
        tty = self.tty if self.tty in [True, False] else not self.remote_command
        args = [env.ssh_term, '-t'] if tty else [env.ssh, '-T']
        if self.port:
            port_flag = '-P' if env.suite is Suite.PUTTY else '-p'
            args.extend([port_flag, self.port])
        if self.cert_file:
            args.extend(['-o CertificateFile={}'.format(self.cert_file)])
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
        if self.remote:
            args.append(self.remote.ToArg())
        if self.remainder:
            args.extend(self.remainder)
        if self.remote_command:
            if env.suite is Suite.OPENSSH:
                args.append('--')
                args.extend(self.remote_command)
            elif tty:
                with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as self._remote_command_file:
                    self._remote_command_file.write(' '.join(self.remote_command))
                args.extend(['-m', self._remote_command_file.name])
            else:
                args.extend(self.remote_command)
        return args

    def Run(self, env=None, putty_force_connect=False, explicit_output_file=None, explicit_error_file=None, explicit_input_file=None):
        """Run the SSH command using the given environment.

    Args:
      env: Environment, environment to run in (or current if None).
      putty_force_connect: bool, whether to inject 'y' into the prompts for
        `plink`, which is insecure and not recommended. It serves legacy
        compatibility purposes for existing usages only; DO NOT SET THIS IN NEW
        CODE.
      explicit_output_file: Pipe stdout into this file-like object
      explicit_error_file: Pipe stderr into this file-like object
      explicit_input_file: Pipe stdin from this file-like object

    Raises:
      MissingCommandError: If SSH command(s) not found.
      CommandError: SSH command failed (not to be confused with the eventual
        failure of the remote command).

    Returns:
      int, The exit code of the remote command, forwarded from the client.
    """
        env = env or Environment.Current()
        args = self.Build(env)
        log.debug('Running command [{}].'.format(' '.join(args)))
        if env.suite is Suite.PUTTY and putty_force_connect:
            in_str = 'y\n'
        else:
            in_str = None
        extra_popen_kwargs = {}
        if explicit_output_file:
            extra_popen_kwargs['stdout'] = explicit_output_file
        if explicit_error_file:
            extra_popen_kwargs['stderr'] = explicit_error_file
        if explicit_input_file:
            extra_popen_kwargs['stdin'] = explicit_input_file
        status = execution_utils.Exec(args, no_exit=True, in_str=in_str, **extra_popen_kwargs)
        if self._remote_command_file:
            try:
                os.remove(self._remote_command_file.name)
            except OSError:
                log.debug('Failed to delete remote command file [{}]'.format(self._remote_command_file.name))
                pass
        if status == env.ssh_exit_code:
            raise CommandError(args[0], return_code=status)
        return status