from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import shlex
import sys
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import platforms
import six
def _OpenCompletionsOutputStream():
    """Returns the completions output stream."""
    return os.fdopen(COMPLETIONS_OUTPUT_FD, 'wb')