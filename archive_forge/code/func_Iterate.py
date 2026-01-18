from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import pkg_resources
import jsonschema
def Iterate(self, parsed_yaml):
    """Validates parsed_yaml against JSON schema and returns all errors.

    Args:
      parsed_yaml: YAML to validate

    Raises:
      ValidationError: if the template doesn't obey the schema.

    Returns:
      A list of all errors, empty if no validation errors.
    """
    return self._validator.iter_errors(parsed_yaml)