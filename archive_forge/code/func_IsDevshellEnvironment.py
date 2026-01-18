from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.core.util import encoding
def IsDevshellEnvironment():
    return bool(encoding.GetEncodedValue(os.environ, DEVSHELL_ENV, False)) or HasDevshellAuth()