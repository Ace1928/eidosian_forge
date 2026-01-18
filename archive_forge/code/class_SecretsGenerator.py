from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import base64
import collections
import json
import os
import os.path
import re
import uuid
from apitools.base.py import encoding_helper
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.app import yaml_parsing as app_engine_yaml_parsing
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import service as k8s_service
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.auth import auth_util
from googlecloudsdk.command_lib.code import builders
from googlecloudsdk.command_lib.code import common
from googlecloudsdk.command_lib.code import dataobject
from googlecloudsdk.command_lib.code import secrets
from googlecloudsdk.command_lib.code import yaml_helper
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.run import secrets_mapping
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
class SecretsGenerator(KubeConfigGenerator):
    """Generate kubernetes secrets for referenced secrets."""

    def __init__(self, service_name, env_secrets, secret_volumes, namespace, allow_secret_manager=_FLAG_UNSPECIFIED):
        self.project_name = properties.VALUES.core.project.Get()
        self.service_name = service_name
        self.secret_volumes = secret_volumes
        self.all_secrets = {}
        secrets_builder = collections.defaultdict(set)
        for _, secret in env_secrets.items():
            secrets_builder[secret.name, secret.mapped_secret].add(secret.key)
        for secret in secret_volumes:
            if not secret.items:
                secrets_builder[secret.secret_name, secret.mapped_secret].add(secrets_mapping.SpecialVersion.MOUNT_ALL)
            else:
                for item in secret.items:
                    secrets_builder[secret.secret_name, secret.mapped_secret].add(item.key)
        for (secret_name, mapped_secret), versions in secrets_builder.items():
            self.all_secrets[secret_name] = secrets.SecretManagerSecret(name=secret_name, versions=frozenset(versions), mapped_secret=mapped_secret)
        self.namespace = namespace
        self.allow_secret_manager = allow_secret_manager

    def CreateConfigs(self):
        if not self.all_secrets:
            return []
        if self.allow_secret_manager is _FLAG_UNSPECIFIED:
            requested = []
            for key, sec in self.all_secrets.items():
                if sec.mapped_secret:
                    requested.append(sec.mapped_secret)
                else:
                    requested.append(key)
            secrets_msg = 'This config references secrets stored in secret manager. Continuing will fetch the secret values and download the secrets to your local machine.'
            prompt_string = 'Fetch secrets from secret manager for {}?'.format(requested)
            if console_io.CanPrompt() and console_io.PromptContinue(message=secrets_msg, prompt_string=prompt_string):
                log.status.Print('You can skip this message in the future by passing the flag --allow-secret-manager')
                self.allow_secret_manager = True
        if not self.allow_secret_manager or self.allow_secret_manager is _FLAG_UNSPECIFIED:
            raise SecretsNotAllowedError('Config requires secrets but access to secret manager was not allowed. Replace secrets with environment variables or allow secret manager with --allow-secret-manager to proceed.')
        return secrets.BuildSecrets(self.project_name, set(self.all_secrets.values()), self.namespace)

    def ModifyDeployment(self, deployment):
        if deployment['metadata']['name'] != self.service_name:
            return
        if not self.secret_volumes:
            return
        volumes = yaml_helper.GetOrCreate(deployment, ('spec', 'template', 'spec', 'volumes'), list)
        for volume in self.secret_volumes:
            _AddSecretVolumeByName(volumes, volume.secret_name, volume.name, volume.items)

    def ModifyContainer(self, container):
        if container['name'] != '{}-container'.format(self.service_name):
            return
        if not self.secret_volumes:
            return
        mounts = yaml_helper.GetOrCreate(container, ('volumeMounts',), list)
        for volume in self.secret_volumes:
            _AddVolumeMount(mounts, volume.name, volume.mount_path)