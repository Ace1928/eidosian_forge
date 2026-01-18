from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import files
import six
from six.moves import configparser
def AllProperties(self):
    """Returns a dictionary of properties in the file."""
    return dict(self._properties)