from __future__ import print_function
from __future__ import unicode_literals
import collections
import contextlib
import gzip
import json
import keyword
import logging
import os
import re
import tempfile
import six
from six.moves import urllib_parse
import six.moves.urllib.error as urllib_error
import six.moves.urllib.request as urllib_request
@contextlib.contextmanager
def Chdir(dirname, create=True):
    if not os.path.exists(dirname):
        if not create:
            raise OSError('Cannot find directory %s' % dirname)
        else:
            os.mkdir(dirname)
    previous_directory = os.getcwd()
    try:
        os.chdir(dirname)
        yield
    finally:
        os.chdir(previous_directory)