from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions
import six
class InvalidFaultConfigurationError(exceptions.Error):
    """Error if a fault configuration is improperly specified."""