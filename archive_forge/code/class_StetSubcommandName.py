from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import shutil
from gslib import storage_url
from gslib.utils import execution_util
from gslib.utils import temporary_file_util
from boto import config
class StetSubcommandName(object):
    """Enum class for available STET subcommands."""
    ENCRYPT = 'encrypt'
    DECRYPT = 'decrypt'