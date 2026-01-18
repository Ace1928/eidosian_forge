from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.domains import registrations
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
def _AddPriceFlagsToParser(parser, mutation_op):
    command = ''
    if mutation_op == MutationOp.REGISTER:
        command = 'get-register-parameters'
    elif mutation_op == MutationOp.TRANSFER:
        command = 'get-transfer-parameters'
    base.Argument('--yearly-price', help='You must accept the yearly price of the domain, either in the interactive flow or using this flag. The expected format is a number followed by a currency code, e.g. "12.00 USD". You can get the price using the {} command.'.format(command)).AddToParser(parser)