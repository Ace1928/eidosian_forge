from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import enum
from googlecloudsdk.command_lib.container.binauthz import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
import six
def GetResourceFileType(file_name):
    _, ext = os.path.splitext(file_name)
    if ext == '.json':
        return ResourceFileType.JSON
    elif ext in ('.yaml', '.yml'):
        return ResourceFileType.YAML
    else:
        return ResourceFileType.UNKNOWN