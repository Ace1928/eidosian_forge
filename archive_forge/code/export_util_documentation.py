from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import files
Writes a message as YAML to a stream.

  Args:
    manifest_dict: parsed yaml definition.
    args: arguments from command line.
  