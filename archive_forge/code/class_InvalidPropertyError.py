from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.feature_flags import config
from googlecloudsdk.core.util import files
class InvalidPropertyError(ValidationBaseError):
    """Raised when a property is not a valid Cloud SDK property."""

    def __init__(self, property_name, reason):
        """Instantiates the InvalidPropertyError class.

    Args:
      property_name: str, name of the property.
      reason: str, reason for the error.
    """
        header = 'INVALID_PROPERTY_ERROR'
        message = '[{}] is not a valid Cloud SDK property. {}'.format(property_name, reason)
        super(InvalidPropertyError, self).__init__(header, message)