from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
def AddModuleEnablementArgument(parser):
    base.ChoiceArgument('--enablement-state', required=True, metavar='ENABLEMENT_STATE', choices=['enabled', 'disabled'], help_str='Module enablement state in Security Command Center').AddToParser(parser)