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
class _SectionContextAware(_Section):
    """Contains the properties for the 'context_aware' section."""

    def __init__(self):
        super(_SectionContextAware, self).__init__('context_aware')
        self.use_client_certificate = self._AddBool('use_client_certificate', help_text='If True, use client certificate to authorize user device using Context-aware access. This includes user login as well. Some services may not support client certificate authorization. If a command sends requests to such services, the client certificate will not be validated. Run `gcloud topic client-certificate` for list of services supporting this feature.', default=False)
        self.always_use_mtls_endpoint = self._AddBool('always_use_mtls_endpoint', help_text='If True, use the mTLS endpoints regardless of the value of context_aware/use_client_certificate.', default=False, hidden=True)
        self.auto_discovery_file_path = self._Add('auto_discovery_file_path', validator=ExistingAbsoluteFilepathValidator, help_text='File path for auto discovery configuration file.', hidden=True)
        self.certificate_config_file_path = self._Add('certificate_config_file_path', validator=ExistingAbsoluteFilepathValidator, help_text='File path for certificate configuration file.', hidden=True)