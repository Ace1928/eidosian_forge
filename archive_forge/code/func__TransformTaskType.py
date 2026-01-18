from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.command_lib.tasks import parsers
def _TransformTaskType(r):
    if _IsPullTask(r):
        return constants.PULL_QUEUE
    if _IsAppEngineTask(r):
        return 'app-engine'
    if _IsHttpTask(r):
        return 'http'