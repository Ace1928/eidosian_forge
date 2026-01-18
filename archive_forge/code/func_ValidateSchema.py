from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.feature_flags import config
from googlecloudsdk.core.util import files
def ValidateSchema(self):
    """Validates the parsed_yaml against the JSON schema at SCHEMA_PATH.

    Returns:
      InvalidSchemaError: If the config file does not match the schema.
    """
    schema_errors = []
    list_of_invalid_schema = yaml_validator.Validator(SCHEMA_PATH).Iterate(self.parsed_yaml)
    for error in list_of_invalid_schema:
        schema_errors.append('{}'.format(error))
    if schema_errors:
        return InvalidSchemaError(invalid_schema_reasons=schema_errors)
    return None