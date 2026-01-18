from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
import sys
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_attr_os
import six
from six.moves import range
from six.moves import urllib
def _UrlEncodeLen(string):
    """Return the length of string when URL-encoded."""
    encoded = urllib.parse.urlencode({'': string})[1:]
    return len(encoded)