from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
import subprocess
import textwrap
from gae_ext_runtime import ext_runtime
from googlecloudsdk.api_lib.app import ext_runtime_adapter
from googlecloudsdk.api_lib.app.images import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
class RubyConfigurator(ext_runtime.Configurator):
    """Generates configuration for a Ruby app."""

    def __init__(self, path, params, ruby_version, entrypoint, packages):
        """Constructor.

    Args:
      path: (str) Root path of the source tree.
      params: (ext_runtime.Params) Parameters passed through to the
        fingerprinters.
      ruby_version: (str) The ruby interpreter in rbenv format
      entrypoint: (str) The entrypoint command
      packages: ([str, ...]) A set of packages to install
    """
        self.root = path
        self.params = params
        self.ruby_version = ruby_version
        self.entrypoint = entrypoint
        self.packages = packages
        if params.deploy:
            self.notify = log.info
        else:
            self.notify = log.status.Print

    def GenerateConfigs(self):
        """Generates all config files for the module.

    Returns:
      (bool) True if files were written.
    """
        all_config_files = []
        if not self.params.appinfo:
            all_config_files.append(self._GenerateAppYaml())
        if self.params.custom or self.params.deploy:
            all_config_files.append(self._GenerateDockerfile())
            all_config_files.append(self._GenerateDockerignore())
        created = [config_file.WriteTo(self.root, self.notify) for config_file in all_config_files]
        if not any(created):
            self.notify('All config files already exist. No files generated.')
        return any(created)

    def GenerateConfigData(self):
        """Generates all config files for the module.

    Returns:
      list(ext_runtime.GeneratedFile):
        The generated files
    """
        if not self.params.appinfo:
            app_yaml = self._GenerateAppYaml()
            app_yaml.WriteTo(self.root, self.notify)
        all_config_files = []
        if self.params.custom or self.params.deploy:
            all_config_files.append(self._GenerateDockerfile())
            all_config_files.append(self._GenerateDockerignore())
        return [f for f in all_config_files if not os.path.exists(os.path.join(self.root, f.filename))]

    def _GenerateAppYaml(self):
        """Generates an app.yaml file appropriate to this application.

    Returns:
      (ext_runtime.GeneratedFile) A file wrapper for app.yaml
    """
        app_yaml = os.path.join(self.root, 'app.yaml')
        runtime = 'custom' if self.params.custom else 'ruby'
        app_yaml_contents = APP_YAML_CONTENTS.format(runtime=runtime, entrypoint=self.entrypoint)
        app_yaml = ext_runtime.GeneratedFile('app.yaml', app_yaml_contents)
        return app_yaml

    def _GenerateDockerfile(self):
        """Generates a Dockerfile appropriate to this application.

    Returns:
      (ext_runtime.GeneratedFile) A file wrapper for Dockerignore
    """
        dockerfile_content = [DOCKERFILE_HEADER]
        if self.ruby_version:
            dockerfile_content.append(DOCKERFILE_CUSTOM_INTERPRETER.format(self.ruby_version))
        else:
            dockerfile_content.append(DOCKERFILE_DEFAULT_INTERPRETER)
        if self.packages:
            dockerfile_content.append(DOCKERFILE_MORE_PACKAGES.format(' '.join(self.packages)))
        else:
            dockerfile_content.append(DOCKERFILE_NO_MORE_PACKAGES)
        dockerfile_content.append(DOCKERFILE_GEM_INSTALL)
        dockerfile_content.append(DOCKERFILE_ENTRYPOINT.format(self.entrypoint))
        dockerfile = ext_runtime.GeneratedFile(config.DOCKERFILE, '\n'.join(dockerfile_content))
        return dockerfile

    def _GenerateDockerignore(self):
        """Generates a .dockerignore file appropriate to this application."""
        dockerignore = os.path.join(self.root, '.dockerignore')
        dockerignore = ext_runtime.GeneratedFile('.dockerignore', DOCKERIGNORE_CONTENTS)
        return dockerignore