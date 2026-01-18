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
def SetInvocationValue(self, prop, value, flag):
    """Set the value of this property for this command, using a flag.

    Args:
      prop: _Property, The property with an explicit value.
      value: str, The value that should be returned while this command is
        running.
      flag: str, The flag that a user can use to set the property, reported if
        it was required at some point but not set by the command line.
    """
    value_flags = self.GetLatestInvocationValues()
    if value:
        prop.Validate(value)
    value_flags[prop] = _Sections._ValueFlag(value, flag)