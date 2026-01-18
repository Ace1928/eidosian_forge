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
class _SectionRunApps(_Section):
    """Contains the properties for the 'runapps' section."""

    def __init__(self):
        super(_SectionRunApps, self).__init__('runapps')
        self.experimental_integrations = self._AddBool('experimental_integrations', help_text='If enabled then the user will have access to integrations that are currently experimental. These integrations will also not beusable in the API for those who are not allowlisted.', default=False, hidden=True)
        self.deployment_service_account = self._Add('deployment_service_account', help_text='Service account to use when deploying integrations.')