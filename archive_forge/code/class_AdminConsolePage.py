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
class AdminConsolePage(validation.Validated):
    """Class representing the admin console page in an `AdminConsole` object."""
    ATTRIBUTES = {URL: _URL_REGEX, NAME: _PAGE_NAME_REGEX}