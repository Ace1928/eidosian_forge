from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.emulators import spanner_util
from googlecloudsdk.command_lib.emulators import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
def _RequireDockerInstalled():
    docker_path = files.FindExecutableOnPath('docker')
    if not docker_path:
        raise DockerNotFoundError('To use the Cloud Spanner Emulator on {platform}, docker must be installed on your system PATH'.format(platform=platforms.OperatingSystem.Current().name))