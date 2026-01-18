from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.workbench import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def GetBootDisk(args, messages):
    """Creates the boot disk config for the instance.

  Args:
    args: Argparse object from Command.Run
    messages: Module containing messages definition for the specified API.

  Returns:
    Boot disk config for the instance.
  """
    boot_disk_message = messages.BootDisk
    boot_disk_encryption_enum = None
    boot_disk_type_enum = None
    kms_key = None
    if args.IsSpecified('boot_disk_type'):
        boot_disk_type_enum = arg_utils.ChoiceEnumMapper(arg_name='boot-disk-type', message_enum=boot_disk_message.DiskTypeValueValuesEnum, include_filter=lambda x: 'UNSPECIFIED' not in x).GetEnumForChoice(arg_utils.EnumNameToChoice(args.boot_disk_type))
    if args.IsSpecified('boot_disk_encryption'):
        boot_disk_encryption_enum = arg_utils.ChoiceEnumMapper(arg_name='boot-disk-encryption', message_enum=boot_disk_message.DiskEncryptionValueValuesEnum, include_filter=lambda x: 'UNSPECIFIED' not in x).GetEnumForChoice(arg_utils.EnumNameToChoice(args.boot_disk_encryption))
    if args.IsSpecified('boot_disk_kms_key'):
        kms_key = args.CONCEPTS.boot_disk_kms_key.Parse().RelativeName()
    return boot_disk_message(diskType=boot_disk_type_enum, diskEncryption=boot_disk_encryption_enum, diskSizeGb=args.boot_disk_size, kmsKey=kms_key)