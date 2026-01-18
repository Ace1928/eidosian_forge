from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import re
import string
import sys
import wsgiref.util
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.api import backendinfo
from googlecloudsdk.third_party.appengine._internal import six_subset
def FixSecureDefaults(self):
    """Forces omitted `secure` handler fields to be set to 'secure: optional'.

    The effect is that `handler.secure` is never equal to the nominal default.
    """
    if self.secure == SECURE_DEFAULT:
        self.secure = SECURE_HTTP_OR_HTTPS