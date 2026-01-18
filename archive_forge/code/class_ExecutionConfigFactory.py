from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
import six
class ExecutionConfigFactory(object):
    """Factory for ExecutionConfig message.

  Add ExecutionConfig related arguments to argument parser and create
  ExecutionConfig message from parsed arguments.
  """

    def __init__(self, dataproc):
        """Factory class for ExecutionConfig message.

    Args:
      dataproc: A api_lib.dataproc.Dataproc instance.
    """
        self.dataproc = dataproc

    def GetMessage(self, args):
        """Builds an ExecutionConfig instance.

    Build a ExecutionConfig instance according to user settings.
    Returns None if all fileds are None.

    Args:
      args: Parsed arguments.

    Returns:
      ExecutionConfig: A ExecutionConfig instance. None if all fields are
      None.
    """
        kwargs = {}
        if args.tags:
            kwargs['networkTags'] = args.tags
        if args.network:
            kwargs['networkUri'] = args.network
        if args.subnet:
            kwargs['subnetworkUri'] = args.subnet
        if args.performance_tier:
            kwargs['performanceTier'] = self.dataproc.messages.ExecutionConfig.PerformanceTierValueValuesEnum(args.performance_tier.upper())
        if args.service_account:
            kwargs['serviceAccount'] = args.service_account
        if args.kms_key:
            kwargs['kmsKey'] = args.kms_key
        if hasattr(args, 'max_idle') and args.max_idle:
            kwargs['idleTtl'] = six.text_type(args.max_idle) + 's'
        if args.ttl:
            kwargs['ttl'] = six.text_type(args.ttl) + 's'
        if args.staging_bucket:
            kwargs['stagingBucket'] = args.staging_bucket
        if not kwargs:
            return None
        return self.dataproc.messages.ExecutionConfig(**kwargs)