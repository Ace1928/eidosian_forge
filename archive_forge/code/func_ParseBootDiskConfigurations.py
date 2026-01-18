from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def ParseBootDiskConfigurations(api_version='v2'):
    """Request hook for parsing boot disk configurations."""

    def Process(unused_ref, args, request):
        """Parses configurations for boot disk.

    Parsing boot disk configuration if --boot-disk flag is set.

    Args:
      unused_ref: ref to the service.
      args:  The args for this method.
      request: The request to be made.

    Returns:
      Request with boot disk configuration fields populated.

    Raises:
      BootDiskConfigurationError: if confidential compute is enable
        but kms-key is not provided.
      BootDiskConfigurationError: if invalid argument name is provided.
    """
        if not args or not args.IsKnownAndSpecified('boot_disk'):
            return request
        kms_key_arg_name = 'kms-key'
        confidential_compute_arg_name = 'confidential-compute'
        for arg_name in args.boot_disk.keys():
            if arg_name not in [kms_key_arg_name, confidential_compute_arg_name]:
                raise BootDiskConfigurationError('--boot-disk only supports arguments: %s and %s' % (confidential_compute_arg_name, kms_key_arg_name))
        tpu_messages = GetMessagesModule(version=api_version)
        enable_confidential_compute = args.boot_disk.get(confidential_compute_arg_name, 'False').lower() == 'true'
        kms_key = args.boot_disk.get(kms_key_arg_name, None)
        if enable_confidential_compute and kms_key is None:
            raise BootDiskConfigurationError('argument --boot-disk: with confidential-compute=%s requires kms-key; received: %s' % (enable_confidential_compute, kms_key))
        customer_encryption_key = tpu_messages.CustomerEncryptionKey(kmsKeyName=kms_key)
        request.node.bootDiskConfig = tpu_messages.BootDiskConfig(customerEncryptionKey=customer_encryption_key, enableConfidentialCompute=enable_confidential_compute)
        return request
    return Process