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
def HasOverriddenEndpoint(api_name):
    """Check if a URL is the result of an endpoint override."""
    try:
        endpoint_override = properties.VALUES.api_endpoint_overrides.Property(api_name).Get()
    except properties.NoSuchPropertyError:
        return False
    return bool(endpoint_override)