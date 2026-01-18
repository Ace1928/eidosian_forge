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
class _SectionSsh(_Section):
    """Contains SSH-related properties."""

    def __init__(self):
        super(_SectionSsh, self).__init__('ssh')
        self.putty_force_connect = self._AddBool('putty_force_connect', default=True, help_text='Whether or not `gcloud` should automatically accept new or changed host keys when executing plink/pscp commands on Windows. Defaults to True, but can be set to False to present these interactive prompts to the user for host key checking.')
        self.verify_internal_ip = self._AddBool('verify_internal_ip', default=True, help_text='Whether or not `gcloud` should perform an initial SSH connection to verify an instance ID is correct when connecting via its internal IP. Without this check, `gcloud` will simply connect to the internal IP of the desired instance, which may be wrong if the desired instance is in a different subnet but happens to share the same internal IP as an instance in the current subnet. Defaults to True.')