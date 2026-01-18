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
class _SectionAccessContextManager(_Section):
    """Contains the properties for the 'access_context_manager' section."""

    def OrganizationValidator(self, org):
        """Checks to see if the Organization string is valid."""
        if org is None:
            return
        if not org.isdigit():
            raise InvalidValueError('The access_context_manager.organization property must be set to the organization ID number, not a string.')

    def __init__(self):
        super(_SectionAccessContextManager, self).__init__('access_context_manager', hidden=True)
        self.policy = self._Add('policy', validator=AccessPolicyValidator, help_text='ID of the policy resource to operate on. Can be found by running the `access-context-manager policies list` command.')
        self.organization = self._Add('organization', validator=self.OrganizationValidator, help_text='Default organization cloud-bindings command group will operate on.')