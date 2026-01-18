from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties as core_properties
from googlecloudsdk.core.resource import config_printer
from googlecloudsdk.core.resource import csv_printer
from googlecloudsdk.core.resource import diff_printer
from googlecloudsdk.core.resource import flattened_printer
from googlecloudsdk.core.resource import json_printer
from googlecloudsdk.core.resource import list_printer
from googlecloudsdk.core.resource import object_printer
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.resource import resource_printer_types as formats
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.resource import table_printer
from googlecloudsdk.core.resource import yaml_printer
class PrinterAttributes(resource_printer_base.ResourcePrinter):
    """Attributes for all printers. This docstring is used to generate topic docs.

  All formats have these attributes.

  Printer attributes:
    disable: Disables formatted output and does not consume the resources.
    json-decode: Decodes string values that are JSON compact encodings of list
      and dictionary objects. This may become the default.
    pager: If True, sends output to a pager.
    private: Disables log file output. Use this for sensitive resource data
      that should not be displayed in log files. Explicit command line IO
      redirection overrides this attribute.
    transforms: Apply projection transforms to the resource values. The default
      is format specific. Use *no-transforms* to disable.
  """