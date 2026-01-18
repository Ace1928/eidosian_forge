from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import functools
import os
import re
import sys
import textwrap
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.configurations import properties_file as prop_files_lib
from googlecloudsdk.core.docker import constants as const_lib
from googlecloudsdk.core.resource import resource_printer_types as formats
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import http_proxy_types
from googlecloudsdk.core.util import scaled_integer
from googlecloudsdk.generated_clients.apis import apis_map
import six
class _SectionApp(_Section):
    """Contains the properties for the 'app' section."""

    def __init__(self):
        super(_SectionApp, self).__init__('app')
        self.promote_by_default = self._AddBool('promote_by_default', help_text='If True, when deploying a new version of a service, that version will be promoted to receive all traffic for the service. This property can be overridden with the `--promote-by-default` or `--no-promote-by-default` flags.', default=True)
        self.stop_previous_version = self._AddBool('stop_previous_version', help_text='If True, when deploying a new version of a service, the previously deployed version is stopped. If False, older versions must be stopped manually.', default=True)
        self.trigger_build_server_side = self._AddBool('trigger_build_server_side', hidden=True, default=None)
        self.use_flex_with_buildpacks = self._AddBool('use_flex_with_buildpacks', hidden=True, default=None)
        self.cloud_build_timeout = self._Add('cloud_build_timeout', validator=_BuildTimeoutValidator, help_text='Timeout, in seconds, to wait for Docker builds to complete during deployments. All Docker builds now use the Cloud Build API.')
        self.container_builder_image = self._Add('container_builder_image', default='gcr.io/cloud-builders/docker', hidden=True)
        self.use_appengine_api = self._AddBool('use_appengine_api', default=True, hidden=True)
        self.num_file_upload_threads = self._Add('num_file_upload_threads', default=None, hidden=True)

        def GetRuntimeRoot():
            sdk_root = config.Paths().sdk_root
            if sdk_root is None:
                return None
            else:
                return os.path.join(config.Paths().sdk_root, 'platform', 'ext-runtime')
        self.runtime_root = self._Add('runtime_root', callbacks=[GetRuntimeRoot], hidden=True)
        self.use_runtime_builders = self._Add('use_runtime_builders', default=None, help_text='If set, opt in/out to a new code path for building applications using pre-fabricated runtimes that can be updated independently of client tooling. If not set, the default path for each runtime is used.')
        self.runtime_builders_root = self._Add('runtime_builders_root', default='gs://runtime-builders/', hidden=True)