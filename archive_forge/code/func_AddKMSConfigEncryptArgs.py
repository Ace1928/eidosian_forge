from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddKMSConfigEncryptArgs(parser):
    """Add args for encrypting volumes under a KMS Config."""
    concept_parsers.ConceptParser([flags.GetKmsConfigPresentationSpec('The KMS Config used to encrypt')]).AddToParser(parser)
    flags.AddResourceAsyncFlag(parser)