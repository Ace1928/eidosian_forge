from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import os
import posixpath
import sys
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
from googlecloudsdk.third_party.appengine.api import client_deployinfo
import six
from six.moves import urllib
def ConvertToCloudRegion(region):
    """Converts a App Engine region to the format used elsewhere in Cloud."""
    if region in {'europe-west', 'us-central'}:
        return region + '1'
    else:
        return region