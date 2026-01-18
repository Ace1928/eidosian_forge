from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
import six
def GetRemoveLabelsFlag(extra_message, labels_name='labels'):
    return base.Argument('--remove-{}'.format(labels_name), metavar='KEY', type=arg_parsers.ArgList(), action=arg_parsers.UpdateAction, help='      List of label keys to remove. If a label does not exist it is\n      silently ignored. If `--update-{labels}` is also specified then\n      `--update-{labels}` is applied first.'.format(labels=labels_name) + extra_message)