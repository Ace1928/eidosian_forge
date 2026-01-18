from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.command_lib.tasks import parsers
def AddListTasksFormats(parser, is_alpha=False):
    parser.display_info.AddTransforms({'tasktype': _TransformTaskType})
    parser.display_info.AddFormat(_ALPHA_TASK_LIST_FORMAT if is_alpha else _TASK_LIST_FORMAT)
    parser.display_info.AddUriFunc(parsers.TasksUriFunc)