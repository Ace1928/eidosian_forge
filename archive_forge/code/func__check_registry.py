from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import six.moves.urllib.parse
def _check_registry(registry):
    parsed_hostname = six.moves.urllib.parse.urlparse('//' + registry)
    if registry != parsed_hostname.netloc:
        raise BadNameException('Invalid registry: %s' % registry)