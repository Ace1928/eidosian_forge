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
def _ReadPublicKeyBytes(self, args):
    try:
        return self._ReadFile(args.public_key_file, max_bytes=65536)
    except files.Error as e:
        raise exceptions.BadFileException('Failed to read public key file [{0}]: {1}'.format(args.public_key_file, e))