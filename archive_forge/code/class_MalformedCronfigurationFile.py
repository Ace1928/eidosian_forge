from __future__ import absolute_import
from __future__ import unicode_literals
import logging
import os
import sys
import traceback
from googlecloudsdk.third_party.appengine._internal import six_subset
class MalformedCronfigurationFile(Exception):
    """Configuration file for Cron is malformed."""
    pass