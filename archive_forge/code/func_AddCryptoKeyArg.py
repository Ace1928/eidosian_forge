from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import googlecloudsdk
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def AddCryptoKeyArg(parser, required=False, hidden=False, with_clear=True):
    """Adds an argument for the crypto key used for CMEK."""
    policy_group = parser
    if with_clear:
        policy_group = parser.add_mutually_exclusive_group(hidden=hidden)
        AddClearCryptoNameArg(policy_group, required, hidden)
    policy_group.add_argument('--crypto-key', required=required, hidden=hidden, help='Fully qualified name of the crypto key to use for customer-managed encryption. If this is unspecified, Google-managed keys will be used for encryption.')