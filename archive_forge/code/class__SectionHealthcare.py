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
class _SectionHealthcare(_Section):
    """Contains the properties for the 'healthcare' section."""

    def __init__(self):
        super(_SectionHealthcare, self).__init__('healthcare')
        self.location = self._Add('location', default='us-central1', help_text='Default location to use when working with Cloud Healthcare  resources. When a `--location` flag is required but not provided, the  command will fall back to this value.')
        self.dataset = self._Add('dataset', help_text='Default dataset to use when working with Cloud Healthcare resources. When a `--dataset` flag is required but not provided, the command will fall back to this value, if set.')