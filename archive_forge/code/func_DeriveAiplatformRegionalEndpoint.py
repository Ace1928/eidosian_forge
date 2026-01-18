from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from six.moves.urllib import parse
def DeriveAiplatformRegionalEndpoint(endpoint, region, is_prediction=False):
    """Adds region as a prefix of the base url."""
    scheme, netloc, path, params, query, fragment = parse.urlparse(endpoint)
    if netloc.startswith('aiplatform'):
        if is_prediction:
            netloc = '{}-prediction-{}'.format(region, netloc)
        else:
            netloc = '{}-{}'.format(region, netloc)
    return parse.urlunparse((scheme, netloc, path, params, query, fragment))