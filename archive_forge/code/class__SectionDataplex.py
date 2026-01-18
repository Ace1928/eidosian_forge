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
class _SectionDataplex(_Section):
    """Contains the properties for the 'dataplex' section."""

    def __init__(self):
        super(_SectionDataplex, self).__init__('dataplex')
        self.location = self._Add('location', help_text='Dataplex location to use. When a `location` is required but not provided by a flag, the command will fall back to this value, if set.')
        self.lake = self._Add('lake', help_text='Dataplex lake to use. When a `lake` is required but not provided by a flag, the command will fall back to this value, if set.')
        self.zone = self._Add('zone', help_text='Dataplex zone to use. When a `zone` is required but not provided by a flag, the command will fall back to this value, if set.')
        self.asset = self._Add('asset', help_text='Dataplex asset to use. When an `asset` is required but not provided by a flag, the command will fall back to this value, if set.')