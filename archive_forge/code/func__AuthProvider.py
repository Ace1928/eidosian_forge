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
def _AuthProvider(name='gcp'):
    """Generate and return an auth provider config.

  Constructs an auth provider config entry readable by kubectl. This tells
  kubectl to call out to a specific gcloud command and parse the output to
  retrieve access tokens to authenticate to the kubernetes master.
  Kubernetes gcp auth provider plugin at
  https://github.com/kubernetes/kubernetes/tree/master/staging/src/k8s.io/client-go/plugin/pkg/client/auth/gcp

  Args:
    name: auth provider name
  Returns:
    dict, valid auth provider config entry.
  Raises:
    Error: Path to sdk installation not found. Please switch to application
    default credentials using one of

    $ gcloud config set container/use_application_default_credentials true
    $ export CLOUDSDK_CONTAINER_USE_APPLICATION_DEFAULT_CREDENTIALS=true.
  """
    provider = {'name': name}
    if name == 'gcp' and (not properties.VALUES.container.use_app_default_credentials.GetBool()):
        bin_name = 'gcloud'
        if platforms.OperatingSystem.IsWindows():
            bin_name = 'gcloud.cmd'
        sdk_bin_path = config.Paths().sdk_bin_path
        if sdk_bin_path is None:
            log.error(SDK_BIN_PATH_NOT_FOUND)
            raise Error(SDK_BIN_PATH_NOT_FOUND)
        cfg = {'cmd-path': os.path.join(sdk_bin_path, bin_name), 'cmd-args': 'config config-helper --format=json', 'token-key': '{.credential.access_token}', 'expiry-key': '{.credential.token_expiry}'}
        provider['config'] = cfg
    return provider