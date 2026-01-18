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
class _SectionBilling(_Section):
    """Contains the properties for the 'auth' section."""
    LEGACY = 'LEGACY'
    CURRENT_PROJECT = 'CURRENT_PROJECT'
    CURRENT_PROJECT_WITH_FALLBACK = 'CURRENT_PROJECT_WITH_FALLBACK'

    def __init__(self):
        super(_SectionBilling, self).__init__('billing')
        self.quota_project = self._Add('quota_project', default=_SectionBilling.CURRENT_PROJECT, help_text=textwrap.dedent('             The Google Cloud project that is billed and charged quota for\n             operations performed in `gcloud`. When unset, the default is\n             [CURRENT_PROJECT]. This default bills and charges quota against the\n             current project. If you need to operate on one project, but need to\n             bill your usage against or use quota from a different project, you\n             can use this flag to specify the billing project. If both\n             `billing/quota_project` and `--billing-project` are specified,\n             `--billing-project` takes precedence.\n             '))