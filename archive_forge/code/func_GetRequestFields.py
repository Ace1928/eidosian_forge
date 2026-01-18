from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.bms.bms_client import BmsClient
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.bms import flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def GetRequestFields(self, args, client, instance):
    return {**super().GetRequestFields(args, client, instance), 'kms_key_version': args.kms_crypto_key_version, 'ssh_keys': args.CONCEPTS.ssh_keys.Parse(), 'clear_ssh_keys': getattr(args, 'clear_ssh_keys', False)}