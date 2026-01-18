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
class _SectionGkebackup(_Section):
    """Contains the properties for 'gkebackup' section."""

    def __init__(self):
        super(_SectionGkebackup, self).__init__('gkebackup')
        self.location = self._Add('location', default='-', help_text='Default location to use when working with Backup for GKE Services resources. When a `--location` flag is required but not provided, the command will fall back to this value.')
        self.backup_plan = self._Add('backup_plan', default='-', help_text='Default backup plan ID to use when working with Backup for GKE Services resources. When a `--backup-plan` flag is required but not provided, the command will fall back to this value.')
        self.backup = self._Add('backup', default='-', help_text='Default backup ID to use when working with Backup for GKE Services resources. When a `--backup` flag is required but not provided, the command will fall back to this value.')
        self.restore = self._Add('restore_plan', default='-', help_text='Default restore plan ID to use when working with Backup for GKE Services resources. When a `--restore-plan` flag is required but not provided, the command will fall back to this value.')
        self.restore = self._Add('restore', default='-', help_text='Default restore ID to use when working with Backup for GKE Services resources. When a `--restore` flag is required but not provided, the command will fall back to this value.')