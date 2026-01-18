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
def GetApiBaseUrlOrThrow(api_name, api_version):
    """Determine base url to use for resources of given version."""
    api_base_url = GetApiBaseUrl(api_name, api_version)
    if api_base_url is None:
        raise UserError('gcloud config property {} needs to be set in api_endpoint_overrides section.'.format(api_name))
    return api_base_url