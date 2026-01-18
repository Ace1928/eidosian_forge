from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.feature_flags import config
from googlecloudsdk.core.util import files
class InvalidOrderError(ValidationBaseError):
    """Raised when the properties are not in alphabetical order."""

    def __init__(self, properties_list):
        """Instantiates the InvalidOrderError class.

    Args:
      properties_list: str, list of all properties in the config file.
    """
        header = 'ALPHABETICAL_ORDER_ERROR'
        message = 'Properties in the Feature Flag Config File must be in alphabetical order:\n\t{properties_list}'.format(properties_list=properties_list)
        super(InvalidOrderError, self).__init__(header, message)