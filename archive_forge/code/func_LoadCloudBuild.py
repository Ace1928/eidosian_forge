from source directory to docker image. They are stored as templated .yaml files
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import enum
import os
import re
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import config as cloudbuild_config
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
import six
import six.moves.urllib.error
import six.moves.urllib.parse
import six.moves.urllib.request
def LoadCloudBuild(self, params):
    """Loads the Cloud Build configuration file for this builder reference.

    Args:
      params: dict, a dictionary of values to be substituted in to the
        Cloud Build configuration template corresponding to this runtime
        version.

    Returns:
      Build message, the parsed and parameterized Cloud Build configuration
        file.

    Raises:
      CloudBuildLoadError: If the Cloud Build configuration file is unknown.
      FileReadError: If reading the configuration file fails.
      InvalidRuntimeBuilderPath: If the path of the configuration file is
        invalid.
    """
    if not self.build_file_uri:
        raise CloudBuildLoadError('There is no build file associated with runtime [{runtime}]'.format(runtime=self.runtime))
    messages = cloudbuild_util.GetMessagesModule()
    with _Read(self.build_file_uri) as data:
        build = cloudbuild_config.LoadCloudbuildConfigFromStream(data, messages=messages, params=params)
    if build.options is None:
        build.options = messages.BuildOptions()
    build.options.substitutionOption = build.options.SubstitutionOptionValueValuesEnum.ALLOW_LOOSE
    for step in build.steps:
        has_yaml_path = False
        has_runtime_version = False
        for env in step.env:
            parts = env.split('=')
            log.debug('Env var in build step: ' + str(parts))
            if 'GAE_APPLICATION_YAML_PATH' in parts:
                has_yaml_path = True
            if 'GOOGLE_RUNTIME_VERSION' in parts:
                has_runtime_version = True
        if not has_yaml_path:
            step.env.append('GAE_APPLICATION_YAML_PATH=${_GAE_APPLICATION_YAML_PATH}')
        if not has_runtime_version and '_GOOGLE_RUNTIME_VERSION' in params:
            step.env.append('GOOGLE_RUNTIME_VERSION=${_GOOGLE_RUNTIME_VERSION}')
    return build