from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import subprocess
import tempfile
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
class EditorException(Error):
    """EditorException is thrown when the editor returns a non-zero exit code."""