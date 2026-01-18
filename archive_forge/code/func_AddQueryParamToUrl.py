from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import binascii
import codecs
import os
import sys
import io
import re
import locale
import collections
import random
import six
import string
from six.moves import urllib
from six.moves import range
from gslib.exception import CommandException
from gslib.lazy_wrapper import LazyWrapper
from gslib.utils.constants import UTF8
from gslib.utils.constants import WINDOWS_1252
from gslib.utils.system_util import IS_CP1252
def AddQueryParamToUrl(url_str, param_name, param_value):
    """Adds a query parameter to a URL string.

  Appends a query parameter to the query string portion of a url. If a parameter
  with the given name was already present, it is not removed; the new name/value
  pair will be appended to the end of the query string. It is assumed that all
  arguments will be of type `str` (either ASCII or UTF-8 encoded) or `unicode`.

  Note that this method performs no URL-encoding. It is the caller's
  responsibility to ensure proper URL encoding of the entire URL; i.e. if the
  URL is already URL-encoded, you should pass in URL-encoded values for
  param_name and param_value. If the URL is not URL-encoded, you should not pass
  in URL-encoded parameters; instead, you could perform URL-encoding using the
  URL string returned from this function.

  Args:
    url_str: (str or unicode) String representing the URL.
    param_name: (str or unicode) String key of the query parameter.
    param_value: (str or unicode) String value of the query parameter.

  Returns:
    (str or unicode) A string representing the modified url, of type `unicode`
    if the url_str argument was a `unicode`, otherwise a `str` encoded in UTF-8.
  """
    scheme, netloc, path, query_str, fragment = urllib.parse.urlsplit(url_str)
    query_params = urllib.parse.parse_qsl(query_str, keep_blank_values=True)
    query_params.append((param_name, param_value))
    new_query_str = '&'.join(['%s=%s' % (k, v) for k, v in query_params])
    new_url = urllib.parse.urlunsplit((scheme, netloc, path, new_query_str, fragment))
    return new_url