from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Dict
from googlecloudsdk.core.resource import custom_printer_base as cp
from surface.run.integrations.types.describe import Params
def _FormatParams(self, params: Params) -> cp.Lines:
    """Formats all the required and optional parameters for an integration.

    Required parameters should come before optional parameters as defined
    in the PRD.

    Args:
      params: Class contains a list of required and optional params.

    Returns:
      custom_printer_base.Lines, formatted output of all the parameters.
    """
    formatted = []
    for param in params.required:
        formatted.append(self._FormatParam(param, 'required'))
    for param in params.optional:
        formatted.append(self._FormatParam(param, 'optional'))
    return cp.Lines(formatted)