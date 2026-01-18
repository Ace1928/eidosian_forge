from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import batch
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import properties
from six.moves.urllib import parse
def _GetBatchUrl(endpoint_url, api_version):
    """Return a batch URL for the given endpoint URL."""
    parsed_endpoint = parse.urlparse(endpoint_url)
    return parse.urljoin('{0}://{1}'.format(parsed_endpoint.scheme, parsed_endpoint.netloc), 'batch/compute/' + api_version)