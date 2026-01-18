from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import property_selector
import six
import six.moves.http_client
def _GetSpecsForVersion(api_version):
    """Get Specs for the given API version.

  This currently always returns _SPECS_V1, but is left here for the future,
  as a pattern for providing different specs for different versions.

  Args:
    api_version: A string identifying the API version, e.g. 'v1'.

  Returns:
    A map associating each message class name with an _InternalSpec object.
  """
    if api_version == 'v1' or api_version == 'v2beta1':
        return _SPECS_V1
    if 'alpha' in api_version:
        return _SPECS_ALPHA
    return _SPECS_BETA