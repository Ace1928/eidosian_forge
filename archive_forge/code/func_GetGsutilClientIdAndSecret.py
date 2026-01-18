from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import errno
import locale
import os
import struct
import sys
import six
from gslib.utils.constants import WINDOWS_1252
def GetGsutilClientIdAndSecret():
    """Returns a tuple of the gsutil OAuth2 client ID and secret.

  Google OAuth2 clients always have a secret, even if the client is an installed
  application/utility such as gsutil.  Of course, in such cases the "secret" is
  actually publicly known; security depends entirely on the secrecy of refresh
  tokens, which effectively become bearer tokens.

  Returns:
    (str, str) A 2-tuple of (client ID, secret).
  """
    if InvokedViaCloudSdk() and CloudSdkCredPassingEnabled():
        return ('32555940559.apps.googleusercontent.com', 'ZmssLNjJy2998hD4CTg2ejr2')
    return ('909320924072.apps.googleusercontent.com', 'p3RlpR10xMFh9ZXBS/ZNLYUu')