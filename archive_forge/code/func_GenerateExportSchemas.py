from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import os
import re
import textwrap
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.resource import yaml_printer
from googlecloudsdk.core.util import files
import six
def GenerateExportSchemas(api, message_name, message_spec, directory=None):
    """Recursively generates export/import YAML schemas for message_spec in api.

  The message and nested messages are generated in separate schema files in the
  current directory. Pre-existing files are silently overwritten.

  Args:
    api: An API registry object.
    message_name: The API message name for message_spec.
    message_spec: An arg_utils.GetRecursiveMessageSpec() message spec.
    directory: The path name of the directory to place the generated schemas,
      None for the current directory.
  """
    ExportSchemasGenerator(api, directory).Generate(message_name, message_spec)