from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
import os
import re
import stat
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
import six
from six.moves import urllib
def add_gcs_scheme_if_missing(url_string):
    """Returns a string with gs:// prefixed, if URL has no scheme."""
    if SCHEME_DELIMITER in url_string:
        return url_string
    return ProviderPrefix.GCS.value + SCHEME_DELIMITER + url_string