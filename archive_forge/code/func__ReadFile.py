from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.kms import flags
from googlecloudsdk.command_lib.kms import maps
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
def _ReadFile(self, path, max_bytes):
    data = files.ReadBinaryFileContents(path)
    if len(data) > max_bytes:
        raise exceptions.BadFileException('The file is larger than the maximum size of {0} bytes.'.format(max_bytes))
    return data