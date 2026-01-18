from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.filestore import filestore_client
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.filestore import flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddKmsKeyArg(parser):
    permission_info = '{} must hold permission {}'.format("The 'Filestore Service Agent' service account", "'Cloud KMS CryptoKey Encrypter/Decrypter'")
    kms_resource_args.AddKmsKeyResourceArg(parser=parser, resource='instance', permission_info=permission_info, required=False)