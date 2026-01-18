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
def _GetClearNamespaceLabelsFlag(resource_type):
    labels_name = 'namespace-labels'
    return calliope_base.Argument('--clear-{}'.format(labels_name), action='store_true', help="          Remove all {resource_type}-level labels from the cluster namespace. If `--update-{labels}` is also specified then\n          `--clear-{labels}` is applied first.\n\n          For example, to remove all labels:\n\n              $ {{command}} {resource_type}_name --clear-{labels}\n\n          To remove all existing {resource_type}-level labels and create two new labels,\n          ``foo'' and ``baz'':\n\n              $ {{command}} {resource_type}_name --clear-{labels} --update-{labels} foo=bar,baz=qux\n          ".format(labels=labels_name, resource_type=resource_type))