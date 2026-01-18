from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def GetRunOptionArg(for_update):
    choices = {'manual': 'The crawler run will have to be triggered manually.', 'scheduled': 'The crawler will run automatically on a schedule.'}
    return base.ChoiceArgument('--run-option', choices=choices, required=not for_update, help_str='Run option of the crawler.')