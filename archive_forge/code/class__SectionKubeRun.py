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
class _SectionKubeRun(_Section):
    """Contains the properties for the 'kuberun' section."""

    def __init__(self):
        super(_SectionKubeRun, self).__init__('kuberun')
        self.enable_experimental_commands = self._AddBool('enable_experimental_commands', help_text='If True, experimental KubeRun commands will not prompt to continue.', hidden=True)
        self.environment = self._Add('environment', help_text='If set, this environment will be used as the deploymenttarget in all KubeRun commands.', hidden=True)
        self.cluster = self._Add('cluster', help_text='ID of the cluster or fully qualified identifier for the cluster', hidden=True)
        self.cluster_location = self._Add('cluster_location', help_text='Zone or region in which the cluster is located.', hidden=True)
        self.use_kubeconfig = self._AddBool('use_kubeconfig', help_text='Use the default or provided kubectl config file.', hidden=True)
        self.kubeconfig = self._Add('kubeconfig', help_text='Absolute path to your kubectl config file.', hidden=True)
        self.context = self._Add('context', help_text='Name of the context in your kubectl config file to use.', hidden=True)