from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import apis_util
from googlecloudsdk.api_lib.util import resource as resource_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import urllib
from six.moves import zip  # pylint: disable=redefined-builtin
import uritemplate
class InvalidGRIPathSyntaxException(GRIException):
    """Exception for when a part of the path of the GRI is syntactically invalid.
  """

    def __init__(self, gri, message):
        super(InvalidGRIPathSyntaxException, self).__init__('The given GRI [{gri}] could not be parsed because the path is invalid: {message}'.format(gri=gri, message=message))