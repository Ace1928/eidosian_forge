from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import re
import stat
import textwrap
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
class ConfigSSH(base.Command):
    """Populate SSH config files with Host entries from each instance.

  *{command}* makes SSHing to virtual machine instances easier
  by adding an alias for each instance to the user SSH configuration
  (`~/.ssh/config`) file.

  In most cases, it is sufficient to run:

    $ {command}

  Each instance will be given an alias of the form
  `NAME.ZONE.PROJECT`. For example, if `example-instance` resides in
  `us-central1-a`, you can SSH to it by running:

    $ ssh example-instance.us-central1-a.MY-PROJECT

  On some platforms, the host alias can be tab-completed, making
  the long alias less daunting to type.

  The aliases created interface with SSH-based programs like
  *scp(1)*, so it is possible to use the aliases elsewhere:

    $ scp ~/MY-FILE example-instance.us-central1-a.MY-PROJECT:~

  Whenever instances are added, removed, or their external IP
  addresses are changed, this command should be re-executed to
  update the configuration.

  This command ensures that the user's public SSH key is present
  in the project's metadata. If the user does not have a public
  SSH key, one is generated using *ssh-keygen(1)* (if the `--quiet`
  flag is given, the generated key will have an empty passphrase).

  ## EXAMPLES
  To populate SSH config file with Host entries from each running instance, run:

    $ {command}

  To remove the change to the SSH config file by this command, run:

    $ {command} --remove
  """
    category = base.TOOLS_CATEGORY

    @staticmethod
    def Args(parser):
        """Set up arguments for this command.

    Args:
      parser: An argparse.ArgumentParser.
    """
        ssh_utils.BaseSSHHelper.Args(parser)
        parser.add_argument('--ssh-config-file', help="        Specifies an alternative per-user SSH configuration file. By\n        default, this is ``{0}''.\n        ".format(ssh.PER_USER_SSH_CONFIG_FILE))
        parser.add_argument('--dry-run', action='store_true', help='If provided, the proposed changes to the SSH config file are printed to standard output and no actual changes are made.')
        parser.add_argument('--remove', action='store_true', help='If provided, any changes made to the SSH config file by this tool are reverted.')

    def GetRunningInstances(self, client):
        """Returns a generator of all running instances in the project."""
        errors = []
        instances = lister.GetZonalResources(service=client.apitools_client.instances, project=properties.VALUES.core.project.GetOrFail(), requested_zones=None, filter_expr='status eq RUNNING', http=client.apitools_client.http, batch_url=client.batch_url, errors=errors)
        if errors:
            utils.RaiseToolException(errors, error_message='Could not fetch all instances:')
        return instances

    def Run(self, args):
        """See ssh_utils.BaseSSHCommand.Run."""
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        ssh_helper = ssh_utils.BaseSSHHelper()
        ssh_helper.Run(args)
        ssh_helper.keys.EnsureKeysExist(args.force_key_file_overwrite, allow_passphrase=True)
        ssh_config_file = files.ExpandHomeDir(args.ssh_config_file or ssh.PER_USER_SSH_CONFIG_FILE)
        instances = None
        try:
            existing_content = files.ReadFileContents(ssh_config_file)
        except files.Error as e:
            existing_content = ''
            log.debug('SSH Config File [{0}] could not be opened: {1}'.format(ssh_config_file, e))
        if args.remove:
            compute_section = ''
            try:
                new_content = _RemoveComputeSection(existing_content)
            except MultipleComputeSectionsError:
                raise MultipleComputeSectionsError(ssh_config_file)
        else:
            ssh_helper.EnsureSSHKeyIsInProject(client, ssh.GetDefaultSshUsername(warn_on_account_user=True), None)
            instances = list(self.GetRunningInstances(client))
            if instances:
                compute_section = _BuildComputeSection(instances, ssh_helper.keys.key_file, ssh.KnownHosts.DEFAULT_PATH)
            else:
                compute_section = ''
        if existing_content and (not args.remove):
            try:
                new_content = _MergeComputeSections(existing_content, compute_section)
            except MultipleComputeSectionsError:
                raise MultipleComputeSectionsError(ssh_config_file)
        elif not existing_content:
            new_content = compute_section
        if args.dry_run:
            log.out.write(new_content or '')
            return
        if new_content != existing_content:
            if os.path.exists(ssh_config_file) and platforms.OperatingSystem.Current() is not platforms.OperatingSystem.WINDOWS:
                ssh_config_perms = os.stat(ssh_config_file).st_mode
                if not (ssh_config_perms & stat.S_IRWXU == stat.S_IWUSR | stat.S_IRUSR and ssh_config_perms & stat.S_IWGRP == 0 and (ssh_config_perms & stat.S_IWOTH == 0)):
                    log.warning('Invalid permissions on [{0}]. Please change to match ssh requirements (see man 5 ssh).')
            files.WriteFileContents(ssh_config_file, new_content, private=True)
        if compute_section:
            log.out.write(textwrap.dedent('          You should now be able to use ssh/scp with your instances.\n          For example, try running:\n\n            $ ssh {alias}\n\n          '.format(alias=_CreateAlias(instances[0]))))
        elif compute_section == '' and instances:
            log.warning('No host aliases were added to your SSH configs because instances have no public IP.')
        elif not instances and (not args.remove):
            log.warning('No host aliases were added to your SSH configs because you do not have any running instances. Try running this command again after running some instances.')