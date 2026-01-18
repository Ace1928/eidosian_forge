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
def _get_scheme_from_url_string(url_string):
    """Returns scheme component of a URL string."""
    end_scheme_idx = url_string.find(SCHEME_DELIMITER)
    if end_scheme_idx == -1:
        return ProviderPrefix.FILE
    else:
        prefix_string = url_string[0:end_scheme_idx].lower()
        if prefix_string not in VALID_SCHEMES:
            raise errors.InvalidUrlError('Unrecognized scheme "{}"'.format(prefix_string))
        return ProviderPrefix(prefix_string)