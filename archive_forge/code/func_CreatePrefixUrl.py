from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
import stat
import sys
from gslib.exception import CommandException
from gslib.exception import InvalidUrlError
from gslib.utils import system_util
from gslib.utils import text_util
def CreatePrefixUrl(self, wildcard_suffix=None):
    prefix = StripOneSlash(self.versionless_url_string)
    if wildcard_suffix:
        prefix = '%s/%s' % (prefix, wildcard_suffix)
    return prefix