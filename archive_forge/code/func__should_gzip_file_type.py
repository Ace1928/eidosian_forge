from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import gzip
import os
import shutil
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def _should_gzip_file_type(gzip_settings, file_path):
    """Determines what, if any, type of file should be gzipped."""
    if not (gzip_settings and file_path):
        return None
    gzip_extensions = gzip_settings.extensions
    if gzip_settings.extensions == user_request_args_factory.GZIP_ALL:
        return gzip_settings.type
    elif isinstance(gzip_extensions, list):
        for extension in gzip_extensions:
            dot_separated_extension = '.' + extension.lstrip(' .')
            if file_path.endswith(dot_separated_extension):
                return gzip_settings.type
    return None