from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import os
import re
import textwrap
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.resource import yaml_printer
from googlecloudsdk.core.util import files
import six
def _GetSchemaFilePath(self, message_name):
    """Returns the schema file path name given the message name."""
    file_path = self._GetSchemaFileName(message_name)
    if self._directory:
        file_path = os.path.join(self._directory, file_path)
    return file_path