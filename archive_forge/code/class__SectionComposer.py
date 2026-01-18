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
class _SectionComposer(_Section):
    """Contains the properties for the 'composer' section."""

    def __init__(self):
        super(_SectionComposer, self).__init__('composer')
        self.location = self._Add('location', help_text='Composer location to use. Each Composer location constitutes an independent resource namespace constrained to deploying environments into Compute Engine regions inside this location. This parameter corresponds to the /locations/<location> segment of the Composer resource URIs being referenced.')