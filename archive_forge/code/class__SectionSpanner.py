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
class _SectionSpanner(_Section):
    """Contains the properties for the 'spanner' section."""

    def __init__(self):
        super(_SectionSpanner, self).__init__('spanner')
        self.instance = self._Add('instance', help_text='Default instance to use when working with Cloud Spanner resources. When an instance is required but not provided by a flag, the command will fall back to this value, if set.', completer='googlecloudsdk.command_lib.spanner.flags:InstanceCompleter')