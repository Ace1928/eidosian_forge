from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import os
from apitools.base.py import encoding
from googlecloudsdk.api_lib.dataproc import constants
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
import six.moves.urllib.parse
def BuildLoggingConfig(self, messages, driver_logging):
    """Build LoggingConfig from parameters."""
    if not driver_logging:
        return None
    value_enum = messages.LoggingConfig.DriverLogLevelsValue.AdditionalProperty.ValueValueValuesEnum
    config = collections.OrderedDict([(key, value_enum(value)) for key, value in driver_logging.items()])
    return messages.LoggingConfig(driverLogLevels=encoding.DictToAdditionalPropertyMessage(config, messages.LoggingConfig.DriverLogLevelsValue))