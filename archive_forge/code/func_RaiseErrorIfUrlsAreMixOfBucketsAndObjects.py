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
def RaiseErrorIfUrlsAreMixOfBucketsAndObjects(urls, recursion_requested):
    """Raises error if mix of buckets and objects adjusted for recursion."""
    if UrlsAreMixOfBucketsAndObjects(urls) and (not recursion_requested):
        raise CommandException('Cannot operate on a mix of buckets and objects.')