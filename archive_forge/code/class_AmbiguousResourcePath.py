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
class AmbiguousResourcePath(Error):
    """Exception for when API path maps to two different resources."""

    def __init__(self, parser1, parser2):
        super(AmbiguousResourcePath, self).__init__('There already exists parser {0} for same path, can not register another one {1}'.format(parser1, parser2))