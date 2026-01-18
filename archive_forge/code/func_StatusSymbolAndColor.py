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
def StatusSymbolAndColor(self, status: str) -> str:
    """Return the color symbol for the status.

    Args:
      status: string, the status.

    Returns:
      The symbol string.
    """
    if status == states.DEPLOYED or status == states.ACTIVE:
        return GetSymbol(SUCCESS)
    if status in (states.PROVISIONING, states.UPDATING, states.NOT_READY):
        return GetSymbol(UPDATING)
    if status == states.MISSING:
        return GetSymbol(MISSING)
    if status == states.FAILED:
        return GetSymbol(FAILED)
    return GetSymbol(DEFAULT)