from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
import os
from apitools.base.protorpclite import messages
from googlecloudsdk.command_lib.anthos.common import file_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.util import files
import six
def ValidateMessageMappingFile(file_data):
    """Mapping file against krm mapping schema.

  Args:
    file_data: YAMLObject, parsed mapping file data.

  Raises:
    IOError: if schema not found in installed resources.
    ValidationError: if the template doesn't obey the schema.
  """
    validator = yaml_validator.Validator(GetMappingSchema())
    validator.ValidateWithDetailedError(file_data)