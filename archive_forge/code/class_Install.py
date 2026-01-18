from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import shutil
import socket
import subprocess
import sys
from googlecloudsdk.api_lib.transfer import agent_pools_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.transfer import creds_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import platforms
from oauth2client import client as oauth2_client
@base.ReleaseTracks(base.ReleaseTrack.GA)
class Install(base.Command):
    """Install Transfer Service agents."""
    detailed_help = {'DESCRIPTION': '      Install Transfer Service agents to enable you to transfer data to or from\n      POSIX filesystems, such as on-premises filesystems. Agents are installed\n      locally on your machine and run inside Docker containers.\n      ', 'EXAMPLES': "      To create an agent pool for your agent, see the\n      `gcloud transfer agent-pools create` command.\n\n      To install an agent that authenticates with your user account credentials\n      and has default agent parameters, run:\n\n        $ {command} --pool=AGENT_POOL\n\n      You will be prompted to run a command to generate a credentials file if\n      one does not already exist.\n\n      To install an agent that authenticates with a service account with\n      credentials stored at '/example/path.json', run:\n\n        $ {command} --creds-file=/example/path.json --pool=AGENT_POOL\n\n      "}

    @staticmethod
    def Args(parser):
        parser.add_argument('--pool', required=True, help='The agent pool to associate with the newly installed agent. When creating transfer jobs, the agent pool parameter will determine which agents are activated.')
        parser.add_argument('--count', type=int, help=COUNT_FLAG_HELP_TEXT)
        parser.add_argument('--creds-file', help=CREDS_FILE_FLAG_HELP_TEXT)
        parser.add_argument('--docker-network', help=DOCKER_NETWORK_HELP_TEXT)
        parser.add_argument('--enable-multipart', action=arg_parsers.StoreTrueFalseAction, help='Split up files and transfer the resulting chunks in parallel before merging them at the destination. Can be used make transfers of large files faster as long as the network and disk speed are not limiting factors. If unset, agent decides when to use the feature.')
        parser.add_argument('--id-prefix', help='An optional prefix to add to the agent ID to help identify the agent.')
        parser.add_argument('--logs-directory', default='/tmp', help='Specify the absolute path to the directory you want to store transfer logs in. If not specified, gcloud transfer will mount your /tmp directory for logs.')
        parser.add_argument('--memlock-limit', default=64000000, type=int, help="Set the agent container's memlock limit. A value of 64000000 (default) or higher is required to ensure that agent versions 1.14 or later have enough locked memory to be able to start.")
        parser.add_argument('--mount-directories', type=arg_parsers.ArgList(), metavar='MOUNT-DIRECTORIES', help=MOUNT_DIRECTORIES_HELP_TEXT)
        parser.add_argument('--proxy', help=PROXY_FLAG_HELP_TEXT)
        parser.add_argument('--s3-compatible-mode', action='store_true', help=S3_COMPATIBLE_HELP_TEXT)
        hdfs_group = parser.add_group(category='HDFS', sort_args=False)
        hdfs_group.add_argument('--hdfs-namenode-uri', help='A URI representing an HDFS cluster including a schema, namenode, and port. Examples: "rpc://my-namenode:8020", "http://my-namenode:9870".\n\nUse "http" or "https" for WebHDFS. If no schema is provided, the CLI assumes native "rpc". If no port is provided, the default is 8020 for RPC, 9870 for HTTP, and 9871 for HTTPS. For example, the input "my-namenode" becomes "rpc://my-namenode:8020".')
        hdfs_group.add_argument('--hdfs-username', help='Username for connecting to an HDFS cluster with simple auth.')
        hdfs_group.add_argument('--hdfs-data-transfer-protection', choices=['authentication', 'integrity', 'privacy'], help='Client-side quality of protection setting for Kerberized clusters. Client-side QOP value cannot be more restrictive than the server-side QOP value.')
        kerberos_group = parser.add_group(category='Kerberos', sort_args=False)
        kerberos_group.add_argument('--kerberos-config-file', help='Path to Kerberos config file.')
        kerberos_group.add_argument('--kerberos-keytab-file', help='Path to a Keytab file containing the user principal specified with the --kerberos-user-principal flag.')
        kerberos_group.add_argument('--kerberos-user-principal', help='Kerberos user principal to use when connecting to an HDFS cluster via Kerberos auth.')
        kerberos_group.add_argument('--kerberos-service-principal', help='Kerberos service principal to use, of the form "<primary>/<instance>". Realm is mapped from your Kerberos config. Any supplied realm is ignored. If not passed in, it will default to "hdfs/<namenode_fqdn>" (fqdn = fully qualified domain name).')

    def Run(self, args):
        if args.count is not None and args.count < 1:
            raise ValueError('Agent count must be greater than zero.')
        project = properties.VALUES.core.project.Get()
        if not project:
            raise ValueError(MISSING_PROJECT_ERROR_TEXT)
        messages = apis.GetMessagesModule('transfer', 'v1')
        if agent_pools_util.api_get(args.pool).state != messages.AgentPool.StateValueValuesEnum.CREATED:
            raise ValueError('Agent pool not found: ' + args.pool)
        creds_file_path = _authenticate_and_get_creds_file_path(args.creds_file)
        log.status.Print('[1/3] Credentials found ✓')
        _check_if_docker_installed()
        log.status.Print('[2/3] Docker found ✓')
        docker_command = _execute_and_return_docker_command(args, project, creds_file_path)
        if args.count is not None:
            _create_additional_agents(args.count, args.id_prefix, docker_command)
        log.status.Print('[3/3] Agent installation complete! ✓')
        log.status.Print(CHECK_AGENT_CONNECTED_HELP_TEXT_FORMAT.format(pool=args.pool, project=project))