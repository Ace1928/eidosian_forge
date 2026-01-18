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
def GetSymbol(status, encoding=None) -> str:
    """Chooses a symbol to be displayed to the console based on the status.

  Args:
    status: str, defined as a constant in this file.  CloudSDK must
      support Python 2 at the moment so we cannot use the actual enum class.
      If the value is not valid or supported then it will return a default
      symbol.

    encoding: str, defined as a constant in this file.  If not provided, the
      encoding will be fetched from the user's console.

  Returns:
    Symbol (str) to be displayed to the console.
  """
    con = console_attr.GetConsoleAttr()
    if encoding is None:
        encoding = _GetEncoding()
    default_symbol = con.Colorize('~', 'blue')
    status_to_symbol = {SUCCESS: con.Colorize(_PickSymbol('✔', '+', encoding), 'green'), UPDATING: con.Colorize(_PickSymbol('…', '.', encoding), 'yellow'), FAILED: con.Colorize('X', 'red'), MISSING: con.Colorize('?', 'yellow'), DEFAULT: default_symbol}
    return status_to_symbol.get(status, default_symbol)