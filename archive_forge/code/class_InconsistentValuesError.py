from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.feature_flags import config
from googlecloudsdk.core.util import files
class InconsistentValuesError(ValidationBaseError):
    """Raised when the values in a property are not of the same type.

  Attributes:
    header: str, general description of the error.
  """

    def __init__(self, values):
        """Instantiates the InconsistentValuesError class.

    Args:
      values: str, list of values in the property with inconsistent values.
    """
        header = 'INCONSISTENT_PROPERTY_VALUES'
        message = 'Value types are not consistent. Ensure the values {} are of the same type.'.format(values)
        super(InconsistentValuesError, self).__init__(header, message)