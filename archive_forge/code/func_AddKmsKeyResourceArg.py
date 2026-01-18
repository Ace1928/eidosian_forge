from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddKmsKeyResourceArg(parser, required=True):
    concept_parsers.ConceptParser.ForResource(name='--kms-key', resource_spec=GetKmsKeyResourceSpec(), group_help='The Cloud KMS (Key Management Service) Crypto Key that will be used', required=required).AddToParser(parser)