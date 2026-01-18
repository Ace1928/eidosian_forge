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
def TransformConfig(self, record: base.Record) -> cp._Marker:
    """Print the config of the integration.

    Args:
      record: integration_printer.Record class that just holds data.

    Returns:
      The printed output.
    """
    if record.metadata and record.metadata.parameters:
        labeled = []
        config_dict = encoding.MessageToDict(record.resource.config) if record.resource.config else {}
        for param in record.metadata.parameters:
            if config_dict.get(param.config_name):
                name = param.label if param.label else param.config_name
                labeled.append((name, config_dict.get(param.config_name)))
        return cp.Labeled(labeled)
    if record.resource.config:
        return cp.Lines([self._PrintAsYaml({'config': record.resource.config})])
    return None