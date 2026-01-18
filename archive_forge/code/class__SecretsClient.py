from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from apitools.base.py import encoding_helper
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.code import kubernetes
from googlecloudsdk.command_lib.run import secrets_mapping
class _SecretsClient(object):
    """Client implementation for calling Secret Manager to fetch secrets."""

    def __init__(self):
        self.secrets_client = apis.GetClientInstance('secretmanager', 'v1')

    def GetSecretData(self, project, secret_name, mapped_secret, version):
        """Retrieve secret from secret manager."""
        if mapped_secret:
            if mapped_secret.startswith('projects/'):
                resource_name = '{}/versions/{}'.format(mapped_secret, version)
            else:
                resource_name = 'projects/{}/secrets/{}/versions/{}'.format(project, mapped_secret, version)
        else:
            resource_name = 'projects/{}/secrets/{}/versions/{}'.format(project, secret_name, version)
        return self.secrets_client.projects_secrets_versions.Access(SECRETS_MESSAGE_MODULE.SecretmanagerProjectsSecretsVersionsAccessRequest(name=resource_name))