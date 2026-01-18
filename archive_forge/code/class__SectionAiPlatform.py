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
class _SectionAiPlatform(_Section):
    """Contains the properties for the command group 'ai_platform' section."""

    def __init__(self):
        super(_SectionAiPlatform, self).__init__('ai_platform')
        self.region = self._Add('region', help_text='Default region to use when working with AI Platform Training and Prediction resources (currently for Prediction only). It is ignored for training resources for now. The value should be either `global` or one of the supported regions. When a `--region` flag is required but not provided, the command will fall back to this value, if set.')