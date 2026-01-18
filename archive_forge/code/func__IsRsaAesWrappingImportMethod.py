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
def _IsRsaAesWrappingImportMethod(self, import_method, messages):
    return import_method in (messages.ImportJob.ImportMethodValueValuesEnum.RSA_OAEP_3072_SHA1_AES_256, messages.ImportJob.ImportMethodValueValuesEnum.RSA_OAEP_4096_SHA1_AES_256, messages.ImportJob.ImportMethodValueValuesEnum.RSA_OAEP_3072_SHA256_AES_256, messages.ImportJob.ImportMethodValueValuesEnum.RSA_OAEP_4096_SHA256_AES_256)