from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Dict
from googlecloudsdk.core.resource import custom_printer_base as cp
from surface.run.integrations.types.describe import Params
def _FormatParam(self, param: Dict[str, str], setting: str) -> cp.Labeled:
    """Formats individual parameter for an integration.

    Example output:
      param1 [required]:
        This is a description of param1.

    Args:
      param: contains keys such as 'name' and 'description'
      setting: is either 'required' or 'optional'

    Returns:
      custom_printer_base.Lines, formatted output of a singular parameter.
    """
    return cp.Labeled([cp.Lines(['{} [{}]'.format(param['name'], setting), cp.Lines([param['description'], ' '])])])