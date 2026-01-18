from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.feature_flags import config
from googlecloudsdk.core.util import files
class InvalidValueError(ValidationBaseError):
    """Raised when a value does not follow the property's validator."""

    def __init__(self, invalid_values):
        """Instantiates the InvalidValueError class.

    Args:
      invalid_values: str, list of values in the section/property that are
        invalid.
    """
        header = 'INVALID_PROPERTY_VALUES'
        message = "The following values are invalid according to the property's validator: {}".format(invalid_values)
        super(InvalidValueError, self).__init__(header, message)