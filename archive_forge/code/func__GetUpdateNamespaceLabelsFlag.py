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
def _GetUpdateNamespaceLabelsFlag(resource_type):
    """Makes a base.Argument for the `--update-namespace-labels` flag."""
    labels_name = 'namespace-labels'
    return calliope_base.Argument('--update-{}'.format(labels_name), metavar='KEY=VALUE', type=arg_parsers.ArgDict(), action=arg_parsers.UpdateAction, help="      List of {resource_type}-level label KEY=VALUE pairs to update in the cluster namespace. If a\n      label exists, its value is modified. Otherwise, a new label is'\n      created.".format(resource_type=resource_type))