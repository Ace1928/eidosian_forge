from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import signal
import sys
import time
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_completer
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import module_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
def _RunStagedProgressTracker(self, args):
    get_bread = progress_tracker.Stage('Getting bread...', key='bread')
    get_pb_and_j = progress_tracker.Stage('Getting peanut butter...', key='pb')
    make_sandwich = progress_tracker.Stage('Making sandwich...', key='make')
    stages = [get_bread, get_pb_and_j, make_sandwich]
    with progress_tracker.StagedProgressTracker('Making sandwich...', stages, success_message='Time to eat!', failure_message='Time to order delivery..!', tracker_id='meta.make_sandwich') as tracker:
        tracker.StartStage('bread')
        time.sleep(0.5)
        tracker.UpdateStage('bread', 'Looking for bread in the pantry')
        time.sleep(0.5)
        tracker.CompleteStage('bread', 'Got some whole wheat bread!')
        tracker.StartStage('pb')
        time.sleep(1)
        tracker.CompleteStage('pb')
        tracker.StartStage('make')
        time.sleep(1)
        tracker.CompleteStage('make')