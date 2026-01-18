from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.feature_flags import config
from googlecloudsdk.core.util import files
def ValidateConfigFile(self):
    """Validates the config file.

    If the config file has any errors, this method compiles them and then
    returns an easy to read sponge log.

    Raises:
      ValidationFailedError: Error raised when validation fails.
    """
    config_file_errors = []
    if self.parsed_yaml is None:
        return
    if not isinstance(self.parsed_yaml, dict):
        config_file_errors.append(InvalidSchemaError(invalid_schema_reasons=['The file content is not in json format']))
        raise ValidationFailedError(self.config_file_path, config_file_errors, {})
    AppendIfNotNone(config_file_errors, self.ValidateAlphabeticalOrder())
    AppendIfNotNone(config_file_errors, self.ValidateSchema())
    config_file_property_errors = {}
    config_file = files.ReadFileContents(self.config_file_path)
    feature_flags_config = config.FeatureFlagsConfig(config_file)
    for section_property in feature_flags_config.properties:
        property_errors = []
        values_list = feature_flags_config.properties[section_property].values
        AppendIfNotNone(property_errors, self.ValidateValueTypes(values_list))
        AppendIfNotNone(property_errors, self.ValidateValues(values_list, section_property))
        if property_errors:
            config_file_property_errors[section_property] = property_errors
    if config_file_errors or config_file_property_errors:
        raise ValidationFailedError(self.config_file_path, config_file_errors, config_file_property_errors)