from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import errno
import io
import os
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.asset import client_util
from googlecloudsdk.command_lib.asset import utils as asset_utils
from googlecloudsdk.command_lib.util.resource_map.declarative import resource_name_translator
from googlecloudsdk.core import exceptions as c_except
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_filter
from googlecloudsdk.core.util import files
import six
def _TryCreateOutputPath(self, path):
    """Try to create output directory if it doesnt exists."""
    directory = os.path.abspath(path.strip())
    try:
        if os.path.isdir(directory) and files.HasWriteAccessInDir(directory):
            return
        if files.HasWriteAccessInDir(os.path.dirname(directory)):
            console_io.PromptContinue('Path {} does not exists. Do you want to create it?'.format(path), default=True, cancel_on_no=True, cancel_string='Export aborted. No files written.')
            files.MakeDir(path)
        else:
            raise OSError(errno.EACCES)
    except ValueError:
        raise ExportPathException('Can not export to path. [{}] is not a directory.'.format(path))
    except OSError:
        raise ExportPathException('Can not export to path [{}]. Ensure that enclosing path exists and is writeable.'.format(path))