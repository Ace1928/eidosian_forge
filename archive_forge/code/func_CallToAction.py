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
def CallToAction(self, record: base.Record) -> Optional[str]:
    """Call to action to use generated environment variables.

    If the resource state is not ACTIVE then the resource is not ready for
    use and the call to action will not be shown.

    It supports simple template value subsitution. Supported keys are:
    %%project%%: the name of the project
    %%region%%: the region
    %%config.X%%: the attribute from Resource's config with key 'X'
    %%status.X%%: the attribute from ResourceStatus' extraDetails with key 'X'

    Args:
      record: integration_printer.Record class that just holds data.

    Returns:
      A formatted string of the call to action message,
      or None if no call to action is required.
    """
    state = str(record.status.state)
    if state != states.ACTIVE or not record.metadata or (not record.metadata.cta):
        return None
    message = record.metadata.cta
    variables = re.findall('%%([\\w.]+)%%', message)
    for variable in variables:
        value = None
        if variable == 'project':
            value = properties.VALUES.core.project.Get(required=True)
        elif variable == 'region':
            value = record.region
        elif variable.startswith('config.'):
            if record.resource and record.resource.config:
                config_key = variable.replace('config.', '')
                res_config = encoding.MessageToDict(record.resource.config)
                value = res_config.get(config_key)
        elif variable.startswith('status.'):
            if record.status and record.status.extraDetails:
                details_key = variable.replace('status.', '')
                res_config = encoding.MessageToDict(record.status.extraDetails)
                value = res_config.get(details_key)
        if value is None:
            value = 'N/A'
        key = '%%{}%%'.format(variable)
        message = message.replace(key, value)
    return message