from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from functools import wraps
import os.path
import random
import re
import shutil
import tempfile
import six
import boto
import gslib.tests.util as util
from gslib.tests.util import unittest
from gslib.utils.constants import UTF8
from gslib.utils.posix_util import NA_ID
from gslib.utils.posix_util import NA_MODE
def MakeRandomTestString(self):
    """Creates a random string of hex characters 8 characters long."""
    return '%08x' % random.randrange(256 ** 4)