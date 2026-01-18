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
class _SectionTransport(_Section):
    """Contains the properties for the 'transport' section."""

    def __init__(self):
        super(_SectionTransport, self).__init__('transport', hidden=True)
        self.disable_requests_override = self._AddBool('disable_requests_override', default=False, hidden=True, help_text='Global switch to turn off using requests as a transport. Users can use it to switch back to the old mode if requests breaks users.')
        self.opt_out_requests = self._AddBool('opt_out_requests', default=False, hidden=True, help_text='A switch to disable requests for a surface or a command group.')