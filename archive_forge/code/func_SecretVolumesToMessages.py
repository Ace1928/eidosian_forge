from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.command_lib.functions import secrets_config
import six
def SecretVolumesToMessages(secret_volumes, messages, normalize_for_v2=False):
    """Converts secrets from dict to cloud function SecretVolume message list.

  Args:
    secret_volumes: Secrets volumes configuration dict. Prefers a sorted ordered
      dict for consistency.
    messages: The GCF messages module to use.
    normalize_for_v2: If set, normalizes the SecretVolumes to the format the
      GCFv2 API expects.

  Returns:
    A list of Cloud Function SecretVolume messages.
  """
    secret_volumes_messages = []
    mount_path_to_secrets = collections.defaultdict(list)
    for secret_volume_key, secret_volume_value in secret_volumes.items():
        mount_path, secret_file_path = secret_volume_key.split(':', 1)
        if normalize_for_v2:
            secret_file_path = re.sub('^/', '', secret_file_path)
        secret_ref = _ParseSecretRef(secret_volume_value)
        mount_path_to_secrets[mount_path].append({'path': secret_file_path, 'project': secret_ref['project'], 'secret': secret_ref['secret'], 'version': secret_ref['version']})
    for mount_path, secrets in sorted(six.iteritems(mount_path_to_secrets)):
        project = secrets[0]['project']
        secret_value = secrets[0]['secret']
        versions = [messages.SecretVersion(path=secret['path'], version=secret['version']) for secret in secrets]
        secret_volumes_messages.append(messages.SecretVolume(mountPath=mount_path, projectId=project, secret=secret_value, versions=versions))
    return secret_volumes_messages