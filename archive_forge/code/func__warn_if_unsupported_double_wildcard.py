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
def _warn_if_unsupported_double_wildcard(self):
    """Log warning if ** use may lead to undefined results."""
    if not self.object_name:
        return
    delimiter_bounded_url = self.delimiter + self.object_name + self.delimiter
    split_url = delimiter_bounded_url.split('{delim}**{delim}'.format(delim=self.delimiter))
    removed_correct_double_wildcards_url_string = ''.join(split_url)
    if '**' in removed_correct_double_wildcards_url_string:
        log.warning('** behavior is undefined if directly preceded or followed by with characters other than / in the cloud and {} locally.'.format(os.sep))