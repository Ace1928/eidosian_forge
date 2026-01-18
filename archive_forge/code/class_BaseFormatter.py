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
class BaseFormatter(abc.ABC):
    """Prints the run Integration in a custom human-readable format."""

    @abc.abstractmethod
    def TransformConfig(self, record: Record) -> cp._Marker:
        """Override to describe the format of the config of the integration."""

    @abc.abstractmethod
    def TransformComponentStatus(self, record: Record) -> cp._Marker:
        """Override to describe the format of the components and status of the integration."""

    def CallToAction(self, record: Record) -> Optional[str]:
        """Override to return call to action message.

    Args:
      record: dict, the integration.

    Returns:
      A formatted string of the call to action message,
      or None if no call to action is required.
    """
        del record
        return None

    def PrintType(self, ctype):
        """Return the type in a user friendly format.

    Args:
      ctype: the type name to be formatted.

    Returns:
      A formatted string.
    """
        return ctype.replace('google_', '').replace('compute_', '').replace('_', ' ').title()

    def GetResourceState(self, resource):
        """Return the state of the top level resource in the integration.

    Args:
      resource: dict, resource status of the integration resource.

    Returns:
      The state string.
    """
        return resource.get('state', states.UNKNOWN)

    def PrintStatus(self, status):
        """Print the status with symbol and color.

    Args:
      status: string, the status.

    Returns:
      The formatted string.
    """
        return '{} {}'.format(self.StatusSymbolAndColor(status), status)

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