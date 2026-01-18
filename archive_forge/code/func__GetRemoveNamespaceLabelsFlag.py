from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import util as cmd_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def _GetRemoveNamespaceLabelsFlag(resource_type):
    labels_name = 'namespace-labels'
    return calliope_base.Argument('--remove-{}'.format(labels_name), metavar='KEY', type=arg_parsers.ArgList(), action=arg_parsers.UpdateAction, help='      List of {resource_type}-level label keys to remove in the cluster namespace. If a label does not exist it is\n      silently ignored. If `--update-{labels}` is also specified then\n      `--update-{labels}` is applied first.\n      '.format(labels=labels_name, resource_type=resource_type))