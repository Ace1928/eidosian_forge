from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.command_lib.functions import secrets_config
import six
def GetSecretsAsDict(secret_env_vars, secret_volumes):
    """Converts secrets from message to flattened secrets configuration dict.

  Args:
    secret_env_vars: list of cloudfunctions_v1|v2alpha|v2beta.SecretEnvVars
    secret_volumes: list of cloudfunctions_v1|v2alpha|v2beta.SecretVolumes

  Returns:
    OrderedDict[str, str], Secrets configuration sorted ordered dict.
  """
    secrets_dict = {}
    if secret_env_vars:
        secrets_dict.update({sev.key: _GetSecretVersionResource(sev.projectId, sev.secret, sev.version) for sev in secret_env_vars})
    if secret_volumes:
        for secret_volume in secret_volumes:
            mount_path = secret_volume.mountPath
            project = secret_volume.projectId
            secret = secret_volume.secret
            if secret_volume.versions:
                for version in secret_volume.versions:
                    secrets_config_key = mount_path + ':' + version.path
                    secrets_config_value = _GetSecretVersionResource(project, secret, version.version)
                    secrets_dict[secrets_config_key] = secrets_config_value
            else:
                secrets_config_key = mount_path + ':/' + secret
                secrets_config_value = _GetSecretVersionResource(project, secret, 'latest')
                secrets_dict[secrets_config_key] = secrets_config_value
    return _CanonicalizedDict(secrets_dict)