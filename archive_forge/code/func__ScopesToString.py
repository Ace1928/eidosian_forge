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
def _ScopesToString(scopes):
    """Converts scope value to a string."""
    if isinstance(scopes, six_subset.string_types):
        return scopes
    else:
        return ' '.join(scopes)