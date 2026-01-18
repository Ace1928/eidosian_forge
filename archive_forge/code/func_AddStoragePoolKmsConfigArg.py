from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.netapp import util as netapp_api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp import util as netapp_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddStoragePoolKmsConfigArg(parser):
    """Adds a --kms-config arg to the given parser."""
    concept_parsers.ConceptParser.ForResource('--kms-config', flags.GetKmsConfigResourceSpec(), 'The KMS config to attach to the Storage Pool.', flag_name_overrides={'location': ''}).AddToParser(parser)