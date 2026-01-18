from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.core.util import encoding
def HasDevshellAuth():
    port = int(encoding.GetEncodedValue(os.environ, DEVSHELL_CLIENT_PORT, 0))
    return port != 0