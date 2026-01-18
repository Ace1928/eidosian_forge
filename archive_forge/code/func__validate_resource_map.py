from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.util import files
import six
def _validate_resource_map(self):
    """Validates resource map against ~/resource_map_schema.yaml."""
    yaml_validator.Validator(self._schema_file_path).Validate(self._resource_map_data)