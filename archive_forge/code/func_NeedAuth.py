from __future__ import absolute_import
import io
import logging
import os
import random
import re
import time
import urllib
import httplib2
from oauth2client import client
from oauth2client import file as oauth2client_file
from oauth2client import tools
from googlecloudsdk.core.util import encoding
from googlecloudsdk.third_party.appengine.tools.value_mixin import ValueMixin
from googlecloudsdk.third_party.appengine._internal import six_subset
def NeedAuth():
    """Marker that we need auth; it'll actually be tried next time around."""
    auth_errors[0] += 1
    logger.debug('Attempting to auth. This is try %s of %s.', auth_errors[0], self.auth_max_errors)
    if auth_errors[0] > self.auth_max_errors:
        RaiseHttpError(url, response_info, response, 'Too many auth attempts.')