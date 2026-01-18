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
def _GetRequiredFields(fields):
    """Returns the list of required field names in fields.

  Args:
    fields: A message spec fields dict.

  Returns:
    The list of required field names in fields.
  """
    required = []
    for name, value in six.iteritems(fields):
        description = value['description']
        if name != 'additionalProperties' and description.startswith(_REQUIRED):
            required.append(name)
    return required