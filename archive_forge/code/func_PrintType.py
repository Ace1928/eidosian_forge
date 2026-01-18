from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
from typing import Optional
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.command_lib.run.integrations.formatters import states
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import custom_printer_base as cp
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages as runapps
def PrintType(self, ctype):
    """Return the type in a user friendly format.

    Args:
      ctype: the type name to be formatted.

    Returns:
      A formatted string.
    """
    return ctype.replace('google_', '').replace('compute_', '').replace('_', ' ').title()