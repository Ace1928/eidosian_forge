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
class _SectionDevshell(_Section):
    """Contains the properties for the 'devshell' section."""

    def __init__(self):
        super(_SectionDevshell, self).__init__('devshell')
        self.image = self._Add('image', hidden=True, default=const_lib.DEFAULT_DEVSHELL_IMAGE)
        self.metadata_image = self._Add('metadata_image', hidden=True, default=const_lib.METADATA_IMAGE)