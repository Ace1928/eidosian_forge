from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import socket
import threading
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import gce_read
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import retry
import six
from six.moves import http_client
from six.moves import urllib_error
def _CheckMemory(self, check_age):
    if not check_age:
        return self.connected
    if self.expiration_time and self.expiration_time >= time.time():
        return self.connected
    return None