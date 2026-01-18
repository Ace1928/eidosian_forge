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
class GRIPathMismatchException(GRIException):
    """Exception for when the path has the wrong number of segments."""

    def __init__(self, gri, params, collection=None):
        super(GRIPathMismatchException, self).__init__('The given GRI [{gri}] does not match the required structure for this resource type. It must match the format: [{format}]'.format(gri=gri, format=':'.join(reversed(params)) + ('::' + collection if collection else '')))