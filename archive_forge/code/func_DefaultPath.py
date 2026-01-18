from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import unicode_literals
import os
from typing import Any
from googlecloudsdk.api_lib.container import kubeconfig as container_kubeconfig
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
@staticmethod
def DefaultPath():
    """Return default path for kubeconfig file."""
    kubeconfig = encoding.GetEncodedValue(os.environ, 'KUBECONFIG')
    if kubeconfig:
        kubeconfig = kubeconfig.split(os.pathsep)[0]
        return os.path.abspath(kubeconfig)
    home_dir = encoding.GetEncodedValue(os.environ, 'HOME')
    if not home_dir and platforms.OperatingSystem.IsWindows():
        home_drive = encoding.GetEncodedValue(os.environ, 'HOMEDRIVE')
        home_path = encoding.GetEncodedValue(os.environ, 'HOMEPATH')
        if home_drive and home_path:
            home_dir = os.path.join(home_drive, home_path)
        if not home_dir:
            home_dir = encoding.GetEncodedValue(os.environ, 'USERPROFILE')
    if not home_dir:
        raise MissingEnvVarError('environment variable {vars} or KUBECONFIG must be set to store credentials for kubectl'.format(vars='HOMEDRIVE/HOMEPATH, USERPROFILE, HOME,' if platforms.OperatingSystem.IsWindows() else 'HOME'))
    return os.path.join(home_dir, '.kube', 'config')