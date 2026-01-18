from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
def RmEnableFile(ve_dir):
    """Remove enable file."""
    os.unlink('{}/{}'.format(ve_dir, ENABLE_FILE))