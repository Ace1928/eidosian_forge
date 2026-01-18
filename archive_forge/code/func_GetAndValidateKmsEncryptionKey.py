from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import ipaddress
import re
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.composer import parsers
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
import six
def GetAndValidateKmsEncryptionKey(args):
    """Validates the KMS key name.

  Args:
    args: list of all the arguments

  Returns:
    string, a fully qualified KMS resource name

  Raises:
    exceptions.InvalidArgumentException: key name not fully specified
  """
    kms_ref = args.CONCEPTS.kms_key.Parse()
    if kms_ref:
        return kms_ref.RelativeName()
    for keyword in ['kms-key', 'kms-keyring', 'kms-location', 'kms-project']:
        if getattr(args, keyword.replace('-', '_'), None):
            raise exceptions.InvalidArgumentException('--kms-key', 'Encryption key not fully specified.')