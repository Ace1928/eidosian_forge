import gzip
import hashlib
import io
import logging
import os
import re
import socket
import sys
import time
import urllib
from googlecloudsdk.core.util import encoding
from googlecloudsdk.third_party.appengine._internal import six_subset
def GetPlatformToken(os_module=os, sys_module=sys, platform=sys.platform):
    """Returns a 'User-agent' token for the host system platform.

  Args:
    os_module, sys_module, platform: Used for testing.

  Returns:
    String containing the platform token for the host system.
  """
    if hasattr(sys_module, 'getwindowsversion'):
        windows_version = sys_module.getwindowsversion()
        version_info = '.'.join((str(i) for i in windows_version[:4]))
        return platform + '/' + version_info
    elif hasattr(os_module, 'uname'):
        uname = os_module.uname()
        return '%s/%s' % (uname[0], uname[2])
    else:
        return 'unknown'