from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import datetime
import io
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import times
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import urllib
def GetTypeDataName(name, type_name='object'):
    """Returns the data name for name of type type_name.

  Args:
    name: The data name.
    type_name: The data type name.

  Returns:
    The data name for name of type type_name.
  """
    return '{name}::{type_name}'.format(name=name, type_name=type_name)