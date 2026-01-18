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
class _SectionEmulator(_Section):
    """Contains the properties for the 'emulator' section.

  This is used to configure emulator properties for pubsub and datastore, such
  as host_port and data_dir.
  """

    def __init__(self):
        super(_SectionEmulator, self).__init__('emulator', hidden=True)
        self.datastore_data_dir = self._Add('datastore_data_dir')
        self.pubsub_data_dir = self._Add('pubsub_data_dir')
        self.datastore_host_port = self._Add('datastore_host_port', default='localhost:8081')
        self.pubsub_host_port = self._Add('pubsub_host_port', default='localhost:8085')
        self.bigtable_host_port = self._Add('bigtable_host_port', default='localhost:8086')