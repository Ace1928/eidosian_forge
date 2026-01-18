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
class KeygenCommand(object):
    """Platform independent SSH client key generation command.

  For OpenSSH, we use `ssh-keygen(1)`. For PuTTY, we use a custom binary.
  Currently, the only supported algorithm is 'rsa'. The command generates the
  following files:
  - `<identity_file>`: Private key, on OpenSSH format (possibly encrypted).
  - `<identity_file>.pub`: Public key, on OpenSSH format.
  - `<identity_file>.ppk`: Unencrypted PPK key-pair, on PuTTY format.

  The PPK-file is only generated from a PuTTY environment, and encodes the same
  private- and public keys as the other files.

  Attributes:
    identity_file: str, path to private key file.
    allow_passphrase: bool, If True, attempt at prompting the user for a
      passphrase for private key encryption, given that the following conditions
      are also true: - Running in an OpenSSH environment (Linux and Mac) -
      Running in interactive mode (from an actual TTY) - Prompts are enabled in
      gcloud
    reencode_ppk: bool, If True, reencode the PPK file if it was generated with
      a bad encoding, instead of generating a new key. This is only valid for
      PuTTY.
    print_cert: bool, If True, ssh-keygen will print certificate information.
  """

    def __init__(self, identity_file, allow_passphrase=True, reencode_ppk=False, print_cert=False):
        """Construct a suite independent `ssh-keygen` command."""
        self.identity_file = identity_file
        self.allow_passphrase = allow_passphrase
        self.reencode_ppk = reencode_ppk
        self.print_cert = print_cert

    def Build(self, env=None):
        """Construct the actual command according to the given environment.

    Args:
      env: Environment, to construct the command for (or current if None).

    Raises:
      MissingCommandError: If keygen command was not found.

    Returns:
      [str], the command args (where the first arg is the command itself).
    """
        env = env or Environment.Current()
        if not env.keygen:
            raise MissingCommandError('Keygen command not found in the current environment.')
        args = [env.keygen]
        if env.suite is Suite.OPENSSH:
            if self.print_cert:
                args.extend(['-L', '-f', self.identity_file])
            else:
                prompt_passphrase = self.allow_passphrase and console_io.CanPrompt()
                if not prompt_passphrase:
                    args.extend(['-N', ''])
                args.extend(['-t', 'rsa', '-f', self.identity_file])
        else:
            if self.reencode_ppk:
                args.append('--reencode-ppk')
            args.append(self.identity_file)
        return args

    def Run(self, env=None, out_func=None):
        """Run the keygen command in the given environment.

    Args:
      env: Environment, environment to run in (or current if None).
      out_func: str->None: A function call with the stdout of ssh-keygen.

    Raises:
      MissingCommandError: Keygen command not found.
      CommandError: Keygen command failed.
    """
        args = self.Build(env)
        log.debug('Running command [{}].'.format(' '.join(args)))
        status = execution_utils.Exec(args, no_exit=True, out_func=out_func)
        if status:
            raise CommandError(args[0], return_code=status)