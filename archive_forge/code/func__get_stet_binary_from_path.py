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
def _get_stet_binary_from_path():
    """Retrieves STET binary from path if available. Python 2 compatible."""
    for path_directory in os.getenv('PATH').split(os.path.pathsep):
        binary_path = os.path.join(path_directory, 'stet')
        if os.path.exists(binary_path):
            return binary_path