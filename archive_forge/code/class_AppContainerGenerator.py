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
class AppContainerGenerator(KubeConfigGenerator):
    """Generate deployment and service for a developer's app."""

    def __init__(self, service_name, image_name, env_vars=None, env_vars_secrets=None, memory_limit=None, cpu_limit=None, cpu_request=None, readiness_probe=False):
        self._service_name = service_name
        self._image_name = image_name
        self._env_vars = env_vars
        self._env_vars_secrets = env_vars_secrets
        self._memory_limit = memory_limit
        self._cpu_limit = cpu_limit
        self._cpu_request = cpu_request
        self._readiness_probe = readiness_probe

    def CreateConfigs(self):
        deployment, container = _CreateDeployment(self._service_name, self._image_name, self._memory_limit, self._cpu_limit, self._cpu_request, self._readiness_probe)
        default_env_vars = {'K_SERVICE': self._service_name, 'K_CONFIGURATION': 'dev', 'K_REVISION': 'dev-0001'}
        _AddEnvironmentVariables(container, default_env_vars)
        if self._env_vars:
            _AddEnvironmentVariables(container, self._env_vars)
        if self._env_vars_secrets:
            _AddSecretEnvironmentVariables(container, self._env_vars_secrets)
        service = CreateService(self._service_name)
        return [deployment, service]