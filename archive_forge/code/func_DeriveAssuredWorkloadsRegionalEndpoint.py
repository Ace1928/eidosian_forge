from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import re
from googlecloudsdk.api_lib.assured import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from six.moves.urllib import parse
def DeriveAssuredWorkloadsRegionalEndpoint(endpoint, region):
    scheme, netloc, path, params, query, fragment = parse.urlparse(endpoint)
    m = re.match(ENV_NETLOC_REGEX_PATTERN, netloc)
    env = m.group(1)
    netloc_suffix = m.group(3)
    if env:
        netloc = '{}{}-{}'.format(env, region, netloc_suffix)
    else:
        netloc = '{}-{}'.format(region, netloc_suffix)
    return parse.urlunparse((scheme, netloc, path, params, query, fragment))