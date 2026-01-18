from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import io
import re
from typing import Any, Optional
from apitools.base.py import encoding
from googlecloudsdk.command_lib.run.integrations.formatters import base
from googlecloudsdk.command_lib.run.integrations.formatters import states
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import custom_printer_base as cp
from googlecloudsdk.core.resource import yaml_printer as yp
def TransformComponentStatus(self, record: base.Record) -> cp._Marker:
    """Print the component status of the integration.

    Args:
      record: dict, the integration.

    Returns:
      The printed output.
    """
    components = []
    comp_statuses = record.status.resourceComponentStatuses if record.status else []
    for r in comp_statuses:
        console_link = r.consoleLink if r.consoleLink else 'n/a'
        state_name = str(r.state).upper() if r.state else 'N/A'
        state_symbol = self.StatusSymbolAndColor(state_name)
        components.append(cp.Lines(['{} ({})'.format(self.PrintType(r.type), r.name), cp.Labeled([('Console link', console_link), ('Resource Status', state_symbol + ' ' + state_name)])]))
    return cp.Labeled(components)