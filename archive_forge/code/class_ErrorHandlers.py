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
class ErrorHandlers(validation.Validated):
    """Class representing error handler directives in application info."""
    ATTRIBUTES = {ERROR_CODE: validation.Optional(_ERROR_CODE_REGEX), FILE: _FILES_REGEX, MIME_TYPE: validation.Optional(str)}