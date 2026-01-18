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
def _validate_scheme(self):
    if not AzureUrl.is_valid_scheme(self.scheme):
        raise errors.InvalidUrlError('Invalid Azure scheme "{}"'.format(self.scheme))