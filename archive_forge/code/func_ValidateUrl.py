from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
import six.moves.urllib.parse
def ValidateUrl(url):
    """Rudimentary url validator.

  Args:
    url: String

  Returns:
    Whether the input string contains both a scheme and a network location. Note
    that this is a very rudimentary validator and does not work on all cases.
    Invalid urls may still pass this check.
  """
    parsed_url = six.moves.urllib.parse.urlsplit(url)
    if not parsed_url.scheme:
        log.error('Invalid URL - The URL must contain a scheme')
        return False
    if not parsed_url.netloc:
        log.error('Invalid URL - The URL must contain a network location')
        return False
    return True