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
class BaseSSHCLIHelper(BaseSSHHelper):
    """Helper class for subcommands that use ssh or scp."""

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command.

    Please add arguments in alphabetical order except for no- or a clear-
    pair for that argument which can follow the argument itself.

    Args:
      parser: An argparse parser that you can use to add arguments that go
          on the command line after this command. Positional arguments are
          allowed.
    """
        super(BaseSSHCLIHelper, BaseSSHCLIHelper).Args(parser)
        parser.add_argument('--dry-run', action='store_true', help='Print the equivalent scp/ssh command that would be run to stdout, instead of executing it.')
        parser.add_argument('--plain', action='store_true', help='        Suppress the automatic addition of *ssh(1)*/*scp(1)* flags. This flag\n        is useful if you want to take care of authentication yourself or\n        use specific ssh/scp features.\n        ')
        parser.add_argument('--strict-host-key-checking', choices=['yes', 'no', 'ask'], help="        Override the default behavior of StrictHostKeyChecking for the\n        connection. By default, StrictHostKeyChecking is set to 'no' the first\n        time you connect to an instance, and will be set to 'yes' for all\n        subsequent connections.\n        ")
        AddSSHKeyExpirationArgs(parser)

    def Run(self, args):
        super(BaseSSHCLIHelper, self).Run(args)
        if not args.plain:
            self.keys.EnsureKeysExist(args.force_key_file_overwrite, allow_passphrase=True)

    def PreliminarilyVerifyInstance(self, instance_id, remote, identity_file, options, putty_force_connect=False):
        """Verify the instance's identity by connecting and running a command.

    Args:
      instance_id: str, id of the compute instance.
      remote: ssh.Remote, remote to connect to.
      identity_file: str, optional key file.
      options: dict, optional ssh options.
      putty_force_connect: bool, whether to inject 'y' into the prompts for
        `plink`, which is insecure and not recommended. It serves legacy
        compatibility purposes for existing usages only; DO NOT SET THIS IN NEW
        CODE.

    Raises:
      ssh.CommandError: The ssh command failed.
      core_exceptions.NetworkIssueError: The instance id does not match.
    """
        if options.get('StrictHostKeyChecking') == 'yes':
            log.debug('Skipping internal IP verification in favor of strict host key checking.')
            return
        if not properties.VALUES.ssh.verify_internal_ip.GetBool():
            log.warning('Skipping internal IP verification connection and connecting to [{}] in the current subnet. This may be the wrong host if the instance is in a different subnet!'.format(remote.host))
            return
        metadata_id_url = 'http://metadata.google.internal/computeMetadata/v1/instance/id'
        remote_command = ['[ `curl "{}" -H "Metadata-Flavor: Google" -q` = {} ] || exit 23'.format(metadata_id_url, instance_id)]
        cmd = ssh.SSHCommand(remote, identity_file=identity_file, options=options, remote_command=remote_command)
        null_in = FileReader(os.devnull)
        null_out = FileWriter(os.devnull)
        null_err = FileWriter(os.devnull)
        return_code = cmd.Run(self.env, putty_force_connect=putty_force_connect, explicit_output_file=null_out, explicit_error_file=null_err, explicit_input_file=null_in)
        if return_code == 0:
            return
        elif return_code == 23:
            raise core_exceptions.NetworkIssueError('Established connection with host {} but was unable to confirm ID of the instance.'.format(remote.host))
        raise ssh.CommandError(cmd, return_code=return_code)