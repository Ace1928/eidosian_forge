from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.ai.docker import utils
from googlecloudsdk.core import config
def _DockerRunOptions(enable_gpu=False, service_account_key=None, cred_mount_path=_DEFAULT_CONTAINER_CRED_KEY_PATH, extra_run_opts=None):
    """Returns a list of 'docker run' options.

  Args:
    enable_gpu: (bool) using GPU or not.
    service_account_key: (bool) path of the service account key to use in host.
    cred_mount_path: (str) path in the container to mount the credential key.
    extra_run_opts: (List[str]) other custom docker run options.
  """
    if extra_run_opts is None:
        extra_run_opts = []
    runtime = ['--runtime', 'nvidia'] if enable_gpu else []
    if service_account_key:
        mount = ['-v', '{}:{}'.format(service_account_key, cred_mount_path)]
    else:
        adc_file_path = config.ADCEnvVariable() or config.ADCFilePath()
        mount = ['-v', '{}:{}'.format(adc_file_path, cred_mount_path)]
    env_var = ['-e', 'GOOGLE_APPLICATION_CREDENTIALS={}'.format(cred_mount_path)]
    return ['--rm'] + runtime + mount + env_var + ['--ipc', 'host'] + extra_run_opts