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
def _get_docker_command(args, project, creds_file_path):
    """Returns docker command from user arguments and generated values."""
    base_docker_command = ['docker', 'run', '--ulimit', 'memlock={}'.format(args.memlock_limit), '--rm', '-d']
    aws_access_key, aws_secret_key = creds_util.get_default_aws_creds()
    if aws_access_key:
        base_docker_command.append('--env')
        base_docker_command.append('AWS_ACCESS_KEY_ID={}'.format(aws_access_key))
    if aws_secret_key:
        base_docker_command.append('--env')
        base_docker_command.append('AWS_SECRET_ACCESS_KEY={}'.format(aws_secret_key))
    if args.docker_network:
        base_docker_command.append('--network={}'.format(args.docker_network))
    expanded_creds_file_path = _expand_path(creds_file_path)
    expanded_logs_directory_path = _expand_path(args.logs_directory)
    root_with_drive = os.path.abspath(os.sep)
    root_without_drive = os.sep
    mount_entire_filesystem = not args.mount_directories or root_with_drive in args.mount_directories or root_without_drive in args.mount_directories
    if mount_entire_filesystem:
        base_docker_command.append('-v=/:/transfer_root')
    else:
        mount_flags = ['-v={}:/tmp'.format(expanded_logs_directory_path), '-v={creds_file_path}:{creds_file_path}'.format(creds_file_path=expanded_creds_file_path)]
        for path in args.mount_directories:
            mount_flags.append('-v={path}:{path}'.format(path=path))
        base_docker_command.extend(mount_flags)
    if args.proxy:
        base_docker_command.append('--env')
        base_docker_command.append('HTTPS_PROXY={}'.format(args.proxy))
    agent_args = ['gcr.io/cloud-ingest/tsop-agent:latest', '--agent-pool={}'.format(args.pool), '--creds-file={}'.format(expanded_creds_file_path), '--hostname={}'.format(socket.gethostname()), '--log-dir={}'.format(expanded_logs_directory_path), '--project-id={}'.format(project)]
    if mount_entire_filesystem:
        agent_args.append('--enable-mount-directory')
    if args.id_prefix:
        if args.count is not None:
            agent_id_prefix = args.id_prefix + '0'
        else:
            agent_id_prefix = args.id_prefix
        agent_args.append('--agent-id-prefix={}'.format(agent_id_prefix))
    _add_docker_flag_if_user_arg_present(args, agent_args)
    if args.s3_compatible_mode:
        agent_args.append('--enable-s3')
    return base_docker_command + agent_args