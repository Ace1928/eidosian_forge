from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import files
import six
from six.moves import configparser
class PropertiesParseError(Error):
    """An exception to be raised when a properties file is invalid."""