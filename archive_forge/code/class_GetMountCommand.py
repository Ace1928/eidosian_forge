from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from argcomplete.completers import FilesCompleter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.cloud_shell import util
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class GetMountCommand(base.Command):
    """Prints a command to mount the Cloud Shell home directory via sshfs."""
    detailed_help = {'DESCRIPTION': '        *{command}* starts your Cloud Shell if it is not already running, then\n        prints out a command that allows you to mount the Cloud Shell home\n        directory onto your local file system using *sshfs*. You must install\n        and run sshfs yourself.\n\n        After mounting the Cloud Shell home directory, any changes you make\n        under the mount point on your local file system will be reflected in\n        Cloud Shell and vice-versa.\n        ', 'EXAMPLES': '        To print a command that mounts a remote directory onto your local file\n        system, run:\n\n            $ {command} REMOTE-DIR\n        '}

    @staticmethod
    def Args(parser):
        util.ParseCommonArgs(parser)
        parser.add_argument('mount_dir', completer=FilesCompleter, help='        Local directory onto which the Cloud Shell home directory should be\n        mounted.\n        ')

    def Run(self, args):
        if platforms.OperatingSystem.IsWindows():
            raise util.UnsupportedPlatform('get-mount-command is not currently supported on Windows')
        else:
            connection_info = util.PrepareEnvironment(args)
            log.Print('sshfs {user}@{host}: {mount_dir} -p {port} -oIdentityFile={key_file} -oStrictHostKeyChecking=no'.format(user=connection_info.user, host=connection_info.host, mount_dir=args.mount_dir, port=connection_info.port, key_file=connection_info.key))