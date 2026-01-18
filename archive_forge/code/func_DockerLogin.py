from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
import os
import subprocess
import sys
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.credentials import store
from googlecloudsdk.core.docker import client_lib
from googlecloudsdk.core.docker import constants
from googlecloudsdk.core.util import files
import six
def DockerLogin(server, username, access_token):
    """Register the username / token for the given server on Docker's keyring."""
    parsed_url = client_lib.GetNormalizedURL(server)
    server = parsed_url.geturl()
    docker_args = ['login']
    docker_args.append('--username=' + username)
    docker_args.append('--password=' + access_token)
    docker_args.append(server)
    docker_p = client_lib.GetDockerProcess(docker_args, stdin_file=sys.stdin, stdout_file=subprocess.PIPE, stderr_file=subprocess.PIPE)
    stdoutdata, stderrdata = docker_p.communicate()
    if docker_p.returncode == 0:
        _SurfaceUnexpectedInfo(stdoutdata, stderrdata)
    else:
        log.error('Docker CLI operation failed:')
        log.out.Print(stdoutdata)
        log.status.Print(stderrdata)
        raise client_lib.DockerError('Docker login failed.')