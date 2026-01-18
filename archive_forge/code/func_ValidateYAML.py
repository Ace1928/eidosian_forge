from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import json
import os
import re
import textwrap
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import encoding as api_encoding
from apitools.base.py import encoding_helper
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.util import encoding
def ValidateYAML(parsed_yaml, schema_path):
    """Validates YAML against JSON schema.

  Args:
    parsed_yaml: YAML to validate
    schema_path: JSON schema file path.

  Raises:
    IOError: if schema not found in installed resources.
    files.Error: if schema file not found.
    ValidationError: if the template doesn't obey the schema.
    SchemaError: if the schema is invalid.
  """
    yaml_validator.Validator(schema_path).Validate(parsed_yaml)