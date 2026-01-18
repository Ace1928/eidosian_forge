from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from six.moves.urllib import parse
def DeriveMLRegionalEndpoint(endpoint, region):
    scheme, netloc, path, params, query, fragment = parse.urlparse(endpoint)
    netloc = '{}-{}'.format(region, netloc)
    return parse.urlunparse((scheme, netloc, path, params, query, fragment))