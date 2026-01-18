from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.pubsub import topics
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.pubsub import flags
from googlecloudsdk.command_lib.pubsub import resource_args
from googlecloudsdk.command_lib.pubsub import util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
def _GetKmsKeyNameFromArgs(args):
    """Parses the KMS key resource name from args.

  Args:
    args: an argparse namespace. All the arguments that were provided to this
      command invocation.

  Returns:
    The KMS CryptoKey resource name for the key specified in args, or None.
  """
    kms_ref = args.CONCEPTS.kms_key.Parse()
    if kms_ref:
        return kms_ref.RelativeName()
    for keyword in ['topic-encryption-key', 'topic-encryption-key-project', 'topic-encryption-key-location', 'topic-encryption-key-keyring']:
        if args.IsSpecified(keyword.replace('-', '_')):
            raise core_exceptions.Error('--topic-encryption-key was not fully specified.')
    return None