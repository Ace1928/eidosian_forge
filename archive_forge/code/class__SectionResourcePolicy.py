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
class _SectionResourcePolicy(_Section):
    """Contains the properties for the 'resource_policy' section."""

    def __init__(self):
        super(_SectionResourcePolicy, self).__init__('resource_policy', hidden=True)
        self.org_restriction_header = self._Add('org_restriction_header', default=None, help_text='Default organization restriction header to use when working with GCP resources. If set, the value must be in JSON format and must contain a comma separated list of authorized GCP organization IDs. The JSON must then be encoded by following the RFC 4648, section 5, specifications. See https://www.rfc-editor.org/rfc/rfc4648#section-5 for more information about base 64 encoding. And visit https://cloud.google.com/resource-manager/docs/organization-restrictions/overview for more information about organization restrictions.')