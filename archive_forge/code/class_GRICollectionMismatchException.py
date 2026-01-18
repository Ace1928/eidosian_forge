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
class GRICollectionMismatchException(GRIException):
    """Exception for when the parsed GRI collection does not match the expected.
  """

    def __init__(self, gri, expected_collection, parsed_collection):
        super(GRICollectionMismatchException, self).__init__('The given GRI [{gri}] could not be parsed because collection [{expected_collection}] was expected but [{parsed_collection}] was provided. Provide a GRI with the correct collection or drop the specified collection.'.format(gri=gri, expected_collection=expected_collection, parsed_collection=parsed_collection))