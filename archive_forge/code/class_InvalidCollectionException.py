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
class InvalidCollectionException(UserError):
    """A command line that was given did not specify a collection."""

    def __init__(self, collection, api_version=None):
        message = 'unknown collection [{collection}]'.format(collection=collection)
        if api_version:
            message += ' for API version [{version}]'.format(version=api_version)
        super(InvalidCollectionException, self).__init__(message)